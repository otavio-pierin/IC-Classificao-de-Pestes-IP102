import os
import torch
import random
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import cv2
from torchvision import transforms, models, datasets, utils
import sys
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification
import matplotlib.pyplot as plt
from skimage import io
from PIL import Image
from torch import nn, optim
from torch.optim import lr_scheduler 
import torch.nn.functional as F
from torchvision.transforms import v2
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import gc # Importante para limpeza de memória

EXP_PATH = './resultados/'
if not os.path.exists(EXP_PATH):
    os.makedirs(EXP_PATH)
DIRETORIO_RAIZ_DATASET = "../../Datasets/ip102_v1.1" 
IMAGES_ROOT = os.path.join(DIRETORIO_RAIZ_DATASET, "images")
TRAIN_FILE = os.path.join(DIRETORIO_RAIZ_DATASET, "train.txt")
VAL_FILE = os.path.join(DIRETORIO_RAIZ_DATASET, "val.txt") 
TEST_FILE = os.path.join(DIRETORIO_RAIZ_DATASET, "test.txt") 
print(f"--- Configuração do Dataset ---")
print(f"Raiz das Imagens: {IMAGES_ROOT}")
print(f"Arquivo de Treino: {TRAIN_FILE}")

def get_class_counts(train_file_path):
    """Lê o train.txt e conta as amostras por classe. Retorna: {idx: contagem}"""
    class_counts = {} 
    print(f"Contando amostras de treino em: {train_file_path}")
    try:
        with open(train_file_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 2: continue
            try:
                class_index = int(parts[1])
                class_counts[class_index] = class_counts.get(class_index, 0) + 1
            except ValueError:
                continue
    except FileNotFoundError:
        print(f"ERRO FATAL: Arquivo de treino '{train_file_path}' não encontrado.")
        sys.exit(1)
    print("Contagem de amostras concluída.")
    return class_counts

def create_class_remapping(class_counts, strategy):
    """
    Cria o dicionário de re-mapeamento.
    Estratégias:
    - 'Full-All': Todas as classes.
    - 'Full-N': Top N classes mais frequentes (ex: Full-10).
    - 'Mid-N': Classes com > 200 img que NÃO estão no Top N (ex: Mid-10 exclui o top 10 e pega o resto > 200).
    """
    print(f"Criando subconjunto de dados para a estratégia: {strategy}")
    
    if strategy.upper() == "FULL-ALL":
        return None

    if "-" not in strategy:
         raise ValueError(f"Estratégia '{strategy}' inválida. Use 'Full-N' ou 'Mid-N'.")

    parts = strategy.split('-')
    mode = parts[0].upper() # FULL ou MID
    n_limit = int(parts[1]) # 'N' (10 ou 20)

    #Ordenar todas as classes por contagem Decrescente
    sorted_classes = sorted(class_counts.items(), key=lambda item: item[1], reverse=True)
    
    target_classes = []

    # --- TOP N (Full-10, Full-20)
    if mode == "FULL":
        target_classes = sorted_classes[:n_limit]
        print(f"Modo FULL: Selecionadas as {len(target_classes)} classes mais frequentes.")

    # --- INTERMEDIÁRIAS (Mid-10, Mid-20)
    elif mode == "MID":
        # 1. Identificar quem são as Top N para excluí-las
        top_n_indices = {item[0] for item in sorted_classes[:n_limit]}
        
        # 2. Filtrar: Não pode ser Top N E tem que ter > 200 imagens
        for c_idx, count in sorted_classes:
            if c_idx not in top_n_indices and count > 200:
                target_classes.append((c_idx, count))
        
        print(f"Modo MID (Exclui Top {n_limit}, Mantém > 200): Selecionadas {len(target_classes)} classes.")

    else:
        raise ValueError("Modo desconhecido. Use 'Full' ou 'Mid'.")

    if not target_classes:
        print("ERRO: Nenhuma classe atendeu aos critérios de filtragem.")
        sys.exit(1)

    # Dicionário de re-mapeamento: {indice_antigo: novo_indice}
    class_remapping = {}
    for new_index, (old_index, count) in enumerate(target_classes):
        class_remapping[old_index] = new_index
        
    return class_remapping

def load_class_names(classes_file_path):
    """
    Lê o arquivo de classes apenas para extrair os nomes.
    """
    class_names = {}   # {0: "Nome Primeira Classe", ...}
    
    print(f"Lendo nomes das classes em: {classes_file_path}")
    
    current_name_parts = []
    current_index_str = None 

    try:
        with open(classes_file_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line: continue

            # Tenta identificar inicio de classe "1. Nome..."
            try:
                parts = line.split()
                if not parts: continue
                
                # Remove ponto se houver "1." -> "1"
                index_part = parts[0].replace('.', '') 
                int(index_part) # Testa se é numero

                # Salva a anterior se existir
                if current_name_parts and current_index_str:
                    idx_0_based = int(current_index_str) - 1
                    class_names[idx_0_based] = " ".join(current_name_parts)

                # Nova classe atual
                current_index_str = index_part
                current_name_parts = parts[1:]

            except ValueError:
                # É continuação do nome ou lixo (ex: ") EC")
                # Se começar com parênteses, é linha de controle antiga, ignoramos
                if line.startswith(")"):
                    continue
                elif current_index_str:
                    current_name_parts.append(line)

        # Salva a última
        if current_name_parts and current_index_str:
            idx_0_based = int(current_index_str) - 1
            class_names[idx_0_based] = " ".join(current_name_parts)

    except Exception as e:
        print(f"Erro ao ler classes: {e}")
        sys.exit(1)
             
    return class_names

# LÓGICA DE MAPEAMENTO
DATASET_STRATEGY = "Full-20"

# Carrega nomes
CLASSES_FILE = os.path.join(DIRETORIO_RAIZ_DATASET, "classes.txt")
all_class_names = load_class_names(CLASSES_FILE)

# Conta amostras
class_counts = get_class_counts(TRAIN_FILE)

# Cria remapeamento
if DATASET_STRATEGY == "Full-All":
    print("Estratégia: Usando o dataset completo.")
    class_remapping = None 
    num_classes = len(all_class_names)
    classes = [all_class_names.get(i, str(i)) for i in range(num_classes)]
else:
    class_remapping = create_class_remapping(class_counts, strategy=DATASET_STRATEGY)
    
    num_classes = len(class_remapping)
    print(f"Número final de classes para o modelo: {num_classes}")
    
    # Gera lista de nomes para o relatório
    classes = ["" for _ in range(num_classes)]
    for old_idx, new_idx in class_remapping.items():
        name = all_class_names.get(old_idx, f"Class {old_idx}")
        classes[new_idx] = name
    
    print(f"Classes selecionadas: {classes}")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('\nDevice: {0}'.format(device))

BATCH_SIZE = 4 
ACCUMULATION_STEPS = 16 
NUM_EPOCHS = 10
LEARNING_RATE = 0.0001
num_workers=4
ARCH_NAME = 'convnext_tiny' 

print(f"modelo: {ARCH_NAME}")
print(f"Batch size: {BATCH_SIZE}")
print(f"ACCUMULATION_STEPS: {ACCUMULATION_STEPS}")
print(f"NUM_EPOCHS: {NUM_EPOCHS}")
print(f"num_workers: {num_workers}")
print(f"Dataset Strategy: {DATASET_STRATEGY}")

def custom_collate_fn(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    if not batch:
        return None, None 
    return torch.utils.data.dataloader.default_collate(batch)


class IP102Dataset(Dataset):
    def __init__(self, list_file_path, images_root, class_remapping, transform=None): 
        self.images_root = images_root
        self.transform = transform
        self.data = []      
        self.labels = []    
        self.class_remapping = class_remapping 

        print(f"Lendo arquivo de lista: {list_file_path}")
        try:
            with open(list_file_path, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"ERRO FATAL: Arquivo de lista não encontrado em {list_file_path}")
            raise

        valid_entries = 0
        skipped_entries = 0
        for i, line in enumerate(lines):
            try:
                parts = line.strip().split() 
                if len(parts) < 2:
                    continue

                relative_img_path = parts[0]
                original_label = int(parts[1]) 

                if self.class_remapping is not None:
                    if original_label not in self.class_remapping:
                        skipped_entries += 1
                        continue 
                    
                    new_label = self.class_remapping[original_label]
                else:
                    new_label = original_label

                img_path = os.path.join(self.images_root, relative_img_path)
                
                self.data.append(img_path)
                self.labels.append(new_label) 
                valid_entries += 1

            except Exception as e:
                print(f"AVISO: Linha {i+1}: Erro ao processar '{line.strip()}'. Erro: {e}")

        print(f"Arquivo de lista lido. Entradas Válidas: {valid_entries}. Entradas Puladas: {skipped_entries}.")
        if not self.data:
            raise ValueError(f"Nenhuma entrada válida encontrada após filtrar {list_file_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"ERRO no __getitem__: Falha ao carregar ou transformar {img_path}. Erro: {e}")
            return None, label 
        return image, label, img_path
    

train_transforms = v2.Compose([
    v2.Resize((224, 224)),
    v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
])

val_transforms = v2.Compose([
    v2.Resize((224, 224)),
    v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
])

test_transforms = v2.Compose([
    v2.Resize((224, 224)),
    v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
])

# Inicialize o CutMix E o Normalize
cutmix = v2.CutMix(num_classes=num_classes, alpha=0.5)
normalize = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

print("Transformações v2 (CutMix e Normalize) inicializadas.")

train_dataset = IP102Dataset(
    list_file_path=TRAIN_FILE, 
    images_root=IMAGES_ROOT, 
    class_remapping=class_remapping, 
    transform=train_transforms
)

val_dataset = IP102Dataset(
    list_file_path=VAL_FILE, 
    images_root=IMAGES_ROOT, 
    class_remapping=class_remapping, 
    transform=val_transforms
)

test_dataset = IP102Dataset(
    list_file_path=TEST_FILE, 
    images_root=IMAGES_ROOT, 
    class_remapping=class_remapping, 
    transform=val_transforms
)

train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=num_workers,
    collate_fn=custom_collate_fn,
    pin_memory=True 
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=num_workers,
    collate_fn=custom_collate_fn, 
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=num_workers,
    collate_fn=custom_collate_fn,
    pin_memory=True
)

#simplenet
class Net(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(in_features=16 * 53 * 53, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1) 
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

def get_model_instance(arch, num_classes):
    """
    Recria a arquitetura do modelo (sem pesos pré-treinados) para 
    que possamos carregar o state_dict (os pesos salvos).
    """
    if arch == 'simplenet':
        m = Net(in_channels=3, num_classes=num_classes)
        
    elif arch == 'resnet50':
        m = models.resnet50(weights=None) 
        num_ftrs = m.fc.in_features
        m.fc = nn.Linear(num_ftrs, num_classes)
        
    elif arch == 'efficientnetb7':
        m = models.efficientnet_b7(weights=None)
        num_ftrs = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(num_ftrs, num_classes)
        
    elif arch == 'vit':
        m = models.vit_b_16(weights=None)
        num_ftrs = m.heads.head.in_features
        m.heads.head = nn.Linear(num_ftrs, num_classes)
    elif arch == 'mobilenet_v3_large':
        m = models.mobilenet_v3_large(weights=None)
        num_ftrs = m.classifier[3].in_features
        m.classifier[3] = nn.Linear(num_ftrs, num_classes)
    elif arch == 'convnext_tiny':
        m = models.convnext_tiny(weights=None)
        num_ftrs = m.classifier[2].in_features
        m.classifier[2] = nn.Linear(num_ftrs, num_classes)
    else:
        raise ValueError(f"Arquitetura '{arch}' não implementada no get_model_instance")
    
    return m

if ARCH_NAME == 'simplenet':
    model = Net(in_channels=3, num_classes=num_classes)
elif ARCH_NAME == 'resnet50':
    model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
elif ARCH_NAME == 'efficientnetb7':
    model = models.efficientnet_b7(weights='EfficientNet_B7_Weights.DEFAULT')
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
elif ARCH_NAME == 'vit':
    model = models.vit_b_16(weights='ViT_B_16_Weights.DEFAULT')
    num_ftrs = model.heads.head.in_features
    model.heads.head = nn.Linear(num_ftrs, num_classes)
elif ARCH_NAME == 'mobilenet_v3_large':
    model = models.mobilenet_v3_large(weights='DEFAULT')
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ftrs, num_classes)
elif ARCH_NAME == 'convnext_tiny':
    model = models.convnext_tiny(weights='DEFAULT')
    num_ftrs = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(num_ftrs, num_classes)

model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS):
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        
        # --- TREINAMENTO ---
        model.train()
        optimizer.zero_grad()
        loss_epoch_train = 0.0
        hits_epoch_train = 0.0
        samples_processed_train = 0.0

        for step, batch_data in enumerate(tqdm(train_loader, desc="Training")):
            if batch_data[0] is None:
                continue

            inputs, labels, _ = batch_data 
            inputs, labels = inputs.to(device), labels.to(device)

            mixed_inputs, mixed_labels = cutmix(inputs, labels) 
            mixed_inputs = normalize(mixed_inputs)

            outputs = model(mixed_inputs)
            log_probs = F.log_softmax(outputs, dim=1)
            loss = -(mixed_labels * log_probs).sum(dim=1).mean()
            
            original_loss_item = loss.item()
            loss = loss / ACCUMULATION_STEPS 

            preds = torch.argmax(outputs, dim=1) 
            loss.backward()
            
            loss_epoch_train += original_loss_item * inputs.size(0)
            hits_epoch_train += (preds == labels).sum().item()
            samples_processed_train += inputs.size(0)

            if (step + 1) % ACCUMULATION_STEPS == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad() 

        epoch_train_loss = loss_epoch_train / samples_processed_train
        epoch_train_acc = hits_epoch_train / samples_processed_train
        print(f"Training Loss: {epoch_train_loss:.4f}, Training Acc: {epoch_train_acc:.4f}")

        # --- VALIDAÇÃO ---
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_data in tqdm(val_loader, desc="Validating"):
                if batch_data[0] is None:
                    continue
                    
                inputs, labels, _ = batch_data
                inputs, labels = inputs.to(device), labels.to(device)
                inputs = normalize(inputs)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        if total > 0:
            val_loss /= total
            val_acc = correct / total
            print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
        else:
            print("Validation: Nenhum dado válido encontrado.")

        scheduler.step()


def test_model(model, test_loader, class_names):
    model.eval()
    
    true_test_list = []
    pred_test_list = []
    prob_test_list = []
    path_test_list = []
    
    print(f"\nIniciando Teste... Salvando resultados em: {EXP_PATH}")

    for i, batch_data in enumerate(tqdm(test_loader, desc="Testing")):
        if batch_data[0] is None:
            continue
            
        img_list, label_list, path_list = batch_data
        img_list = img_list.to(device)
        label_list = label_list.to(device)
        img_list = normalize(img_list)

        with torch.no_grad():
            outputs = model(img_list)
            preds = torch.argmax(outputs, dim=1)
            outputs_prob = F.softmax(outputs, dim=1)
            
            true_test_batch = label_list.cpu().tolist()
            pred_test_batch = preds.cpu().tolist()
            prob_test_batch = outputs_prob.cpu().tolist()
            
            true_test_list.extend(true_test_batch)
            pred_test_list.extend(pred_test_batch)
            prob_test_list.extend(prob_test_batch)
            path_test_list.extend(path_list)

    print('\nCalculando Matriz de Confusão...')
    conf_mat_test = metrics.confusion_matrix(true_test_list, pred_test_list)
    print('\nConfusion matrix (test set)')
    print(conf_mat_test)

    print('\nGerando Relatório de Classificação...')
    target_names = [str(c) for c in class_names]
    
    class_rep_test = metrics.classification_report(
        true_test_list, 
        pred_test_list, 
        target_names=target_names, 
        digits=4,
        zero_division=0
    )
    print('\nClass. report (test set)')
    print(class_rep_test)

    acc_test = metrics.accuracy_score(true_test_list, pred_test_list)
    print('\nValidation Acc.: {:.4f}'.format(acc_test))

    class_rep_path = os.path.join(EXP_PATH, f'{DATASET_STRATEGY}_seed42/classification_report_test_{ARCH_NAME}_{DATASET_STRATEGY}.txt')
    os.makedirs(os.path.dirname(class_rep_path), exist_ok=True) # Garante que a pasta existe
    
    with open(class_rep_path, 'w') as file_rep:
        file_rep.write('\nTEST. SET:')
        file_rep.write('\n\nConfusion matrix:\n')
        file_rep.write(str(conf_mat_test))
        file_rep.write('\n\nClassification report:\n')
        file_rep.write(class_rep_test)
        file_rep.write(f'\n\nAccuracy:\t {acc_test:.4f}')
    print(f"Relatório TXT salvo em: {class_rep_path}")

    csv_path = os.path.join(EXP_PATH, f'{DATASET_STRATEGY}_seed42/class_report_detailed_test_{ARCH_NAME}_{DATASET_STRATEGY}.csv')
    print("Gerando CSV detalhado...")
    
    with open(csv_path, 'w') as file_rep:
        file_rep.write(f'Modelo utilizado:{ARCH_NAME}\n')
        file_rep.write(f'Estratégia utilizada:{DATASET_STRATEGY}\n')
        file_rep.write('#;Image path;Target;Prediction;Correct?\n')
        for class_name in target_names:
            file_rep.write(f';{class_name}')
        
        for i, (path, true, pred, probs) in enumerate(zip(path_test_list, true_test_list, pred_test_list, prob_test_list)):
            is_correct = (true == pred)
            file_rep.write(f'\n{i};{path};{true};{pred};{is_correct}')
            for prob in probs:
                file_rep.write(f';{prob:.4f}')
                
    print(f"CSV detalhado salvo em: {csv_path}")    

# ==============================================================================
# CÓDIGO PARA O ENSEMBLE
# ==============================================================================

def get_features_sequentially(model_configs, dataloader, device, num_classes):
    """
    Carrega um modelo por vez, extrai as probabilidades e libera a memória.
    Retorna:
        X_meta: Matriz [N_amostras, N_modelos * N_classes]
        y_true: Vetor [N_amostras] com os labels reais
        paths_list: Lista com os caminhos das imagens (para o CSV)
    """
    print("\n--- Iniciando Extração Sequencial de Features ---")
    
    all_models_probs = [] # Lista para guardar as matrizes de prob de cada modelo
    y_true = None
    paths_list = [] # Para guardar os caminhos das imagens na primeira passada
    
    # Itera sobre a configuração (Caminho, Arquitetura)
    for i, (path, arch) in enumerate(model_configs):
        print(f"[{i+1}/{len(model_configs)}] Processando: {arch}...")
        
        #Carregar Modelo
        try:
            model = get_model_instance(arch, num_classes)
            model.load_state_dict(torch.load(path))
            model.to(device)
            model.eval()
        except Exception as e:
            print(f"ERRO ao carregar {path}: {e}")
            sys.exit(1)
            
        #Extrair Probabilidades (Inferência no Dataloader inteiro)
        model_probs = []
        collect_labels = (y_true is None) # Só coleta labels/paths na primeira vez
        
        with torch.no_grad():
            for batch_data in tqdm(dataloader, desc=f"Inferência {arch}"):
                if batch_data[0] is None: continue
                
                inputs, labels, paths = batch_data
                inputs = inputs.to(device)
                inputs = normalize(inputs)
                
                outputs = model(inputs)
                # Softmax para garantir que somam 1
                probs = F.softmax(outputs, dim=1).cpu().numpy() 
                
                model_probs.append(probs)
                
                if collect_labels:
                    # Se labels for tensor, converte. Se for lista, mantém.
                    if isinstance(labels, torch.Tensor):
                        current_labels = labels.cpu().numpy()
                    else:
                        current_labels = np.array(labels)
                    
                    y_true = current_labels if y_true is None else np.concatenate([y_true, current_labels])    
                    paths_list.extend(paths)

        #Empilha os batches deste modelo
        # Resultado: [N_amostras, N_classes]
        full_model_probs = np.vstack(model_probs)
        all_models_probs.append(full_model_probs)
        
        #LIMPEZA DE MEMÓRIA
        del model
        torch.cuda.empty_cache()
        gc.collect() # Força o Garbage Collector do Python
        print(f" -> Modelo removido da memória.")

    #Concatena as features de todos os modelos horizontalmente
    # Resultado: [N_amostras, N_modelos * N_classes]
    # Usa float32 para economizar RAM (padrão numpy é float64)
    X_meta = np.hstack(all_models_probs).astype(np.float32)
    
    return X_meta, y_true, paths_list

def train_meta_learner(model_configs, val_loader, method='stacking'):
    """
    Treina um classificador scikit-learn usando extração sequencial.
    """
    print(f"\n--- Treinando Meta-Learner ({method.upper()}) ---")
    
    #Gerar dados (agora passamos configs, não modelos carregados)
    X_val, y_val, _ = get_features_sequentially(model_configs, val_loader, device, num_classes)
    
    #Escolher o meta-modelo
    if method == 'stacking':
        meta_model = LogisticRegression(multi_class='multinomial', max_iter=1000)
    elif method == 'random_forest':
        meta_model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif method == 'gradient_boosting':
        # HistGradientBoostingClassifier é mais rápido e eficiente que GradientBoostingClassifier
        meta_model = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, random_state=42, verbose=1)
    elif method == 'bagging':
        meta_model = BaggingClassifier(n_estimators=50, random_state=42, n_jobs=1) # n_jobs=1 evita deadlock
    else:
        raise ValueError("Método desconhecido para treinamento.")
        
    #Treinar
    print(f"Ajustando {method} com input shape {X_val.shape}...")
    meta_model.fit(X_val, y_val)
    print("Meta-Learner treinado com sucesso.")
    
    return meta_model

def test_ensemble_advanced(model_configs, test_loader, class_names, method='soft', meta_model=None):
    """
    Executa o teste do ensemble com lógica sequencial.
    """
    # Validação
    methods_training_required = ['stacking', 'random_forest', 'gradient_boosting', 'bagging']
    if method in methods_training_required and meta_model is None:
        raise ValueError(f"O método '{method}' exige um 'meta_model' treinado. Rode train_meta_learner primeiro.")

    print(f"\nIniciando Teste Ensemble: {method.upper()}")
    
    #Extrair TODAS as features de uma vez (Sequencialmente)
    print("Gerando features de teste (Isso pode demorar, mas economiza GPU)...")
    X_test, y_test, path_test_list = get_features_sequentially(model_configs, test_loader, device, num_classes)
    
    true_test_list = y_test.tolist()
    
    #Realizar a Predição Final
    if method in methods_training_required:
        # Métodos de Aprendizado (sklearn) usam a matriz X_test direta
        print(f"Realizando predição com {method}...")
        preds = meta_model.predict(X_test)
        pred_test_list = preds.tolist()
        
    else:
        # Métodos Heurísticos (Soft/Hard/Max)
        # X_test shape: [N_Amostras, N_Modelos * N_Classes]
        
        num_models = len(model_configs)
        n_samples = X_test.shape[0]
        n_classes_inner = X_test.shape[1] // num_models
        
        # Reshape para [N_Amostras, N_Modelos, N_Classes]
        # Convertemos para tensor para usar as funções torch (mode, max, etc)
        stacked_probs = torch.tensor(X_test.reshape(n_samples, num_models, n_classes_inner))
        
        if method == 'soft':
            # Média entre os modelos (dim=1 agora, pois é [Samples, Models, Classes])
            avg_probs = torch.mean(stacked_probs, dim=1) 
            preds = torch.argmax(avg_probs, dim=1)
            
        elif method == 'hard':
            # Argmax nas classes (dim=2)
            model_preds = torch.argmax(stacked_probs, dim=2) # [Samples, Models]
            # Moda entre os modelos (dim=1)
            preds, _ = torch.mode(model_preds, dim=1)
            
        elif method == 'max':
            # Max entre os modelos
            max_probs, _ = torch.max(stacked_probs, dim=1)
            preds = torch.argmax(max_probs, dim=1)
            
        pred_test_list = preds.tolist()

    # --- Relatórios ---
    print('\nCalculando Métricas...')
    target_names = [str(c) for c in class_names]
    
    print(metrics.classification_report(true_test_list, pred_test_list, target_names=target_names, zero_division=0))
    acc = metrics.accuracy_score(true_test_list, pred_test_list)
    print(f"Acurácia Final ({method}): {acc:.4f}")
    
    # Salvar TXT
    out_file_txt = os.path.join(EXP_PATH, f'{DATASET_STRATEGY}_seed42/report_ensemble_{method}_top2.txt')
    os.makedirs(os.path.dirname(out_file_txt), exist_ok=True)
    
    with open(out_file_txt, 'w') as f:
        f.write(metrics.classification_report(true_test_list, pred_test_list, target_names=target_names, zero_division=0))
        f.write(f"\nAcc: {acc:.4f}")
        
    # Salvar CSV
    csv_path = os.path.join(EXP_PATH, f'{DATASET_STRATEGY}_seed42/class_report_detailed_test_ensemble_{method}_top2.csv')
    with open(csv_path, 'w') as file_rep:
        file_rep.write(f'Modelo utilizado:ENSEMBLE_{method}\n')
        file_rep.write('#;Image path;Target;Prediction;Correct?\n')
        
        # Iterar
        for i, (path, true, pred) in enumerate(zip(path_test_list, true_test_list, pred_test_list)):
            is_correct = (true == pred)
            file_rep.write(f'{i};{path};{true};{pred};{is_correct}\n')
            
    print(f"CSV salvo em: {csv_path}")

# ==============================================================================
# CONTROLE DE FLUXO PRINCIPAL
# ==============================================================================

# Opções: 'TRAIN' ou 'ENSEMBLE'
EXECUTION_MODE = 'ENSEMBLE' 

# Configurações do Ensemble (só usadas se MODE for 'ENSEMBLE')
ENSEMBLE_METHOD = 'hard' # 'soft', 'hard', 'stacking', 'random_forest', 'gradient_boosting', 'bagging'.

if EXECUTION_MODE == 'TRAIN':
    print(f"\n>>> MODO: TREINAMENTO (Seed: {SEED}) <<<")
    
    #Treina
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS)
    
    #Salva
    save_name = f"ip102_model_{ARCH_NAME}_seed{SEED}.pth"
    torch.save(model.state_dict(), save_name)
    print(f"Modelo salvo como: {save_name}")
    
    #Teste individual
    test_model(model, test_loader, classes)

elif EXECUTION_MODE == 'ENSEMBLE':
    print(f"\n>>> MODO: ENSEMBLE ({ENSEMBLE_METHOD}) <<<")
    
    # LISTA DE CONFIGURAÇÕES: (CAMINHO_DO_ARQUIVO, ARQUITETURA)
    # Verifica se os arquivos .pth realmente existem nos caminhos abaixo!
    MODEL_CONFIGS = [
        (f"./resultados/{DATASET_STRATEGY}_seed42/ip102_model_vit_seed42.pth", "vit"),
        (f"./resultados/{DATASET_STRATEGY}_seed42/ip102_model_convnext_tiny_seed42.pth", "convnext_tiny")
    ]
    
    print(f"Configuração definida para {len(MODEL_CONFIGS)} modelos.")
    print("Os modelos serão carregados sequencialmente para economizar memória.")

    #Treinar Meta-Learner (se necessário)
    meta_learner = None
    if ENSEMBLE_METHOD in ['stacking', 'random_forest', 'gradient_boosting', 'bagging']:
        # Passamos MODEL_CONFIGS direto, sem carregar modelos antes
        meta_learner = train_meta_learner(MODEL_CONFIGS, val_loader, method=ENSEMBLE_METHOD)

    #Rodar Teste do Ensemble
    # Passamos MODEL_CONFIGS direto
    test_ensemble_advanced(
        model_configs=MODEL_CONFIGS, 
        test_loader=test_loader, 
        class_names=classes, 
        method=ENSEMBLE_METHOD,
        meta_model=meta_learner
    )

else:
    print("Modo de execução inválido.")