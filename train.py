import torch
import torchvision.models as models
from torchvision.models import resnet50, ResNet50_Weights,resnet34,ResNet34_Weights,resnet18,ResNet18_Weights
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import random
import glob
import pandas as pd
from tqdm import tqdm
from torchvision.utils import make_grid
import cv2
import sys
import csv
# Grad-CAM
from gradcam.utils import visualize_cam
from gradcam import GradCAM, GradCAMpp

#検出率recall,precision
from sklearn.metrics import recall_score, precision_score, accuracy_score
#混同行列
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn import metrics

def n2p(a,b):
    return os.path.join(a,b)
def mkdir(a):
    if not os.path.exists(a):
        os.mkdir(a)

TRAIN=True

TEST=False

GRAD=False

MODEL=False

PRcurve= False

cross_validation = False

WEIGHTS_DIR = './weights/re_training/resnet34/' #resnet,vit,vggの重みを指定
batch_size = 2#バッチサイズの指定
liver_classes = ["0", "1"]#クラス名
num_epochs = 100#エポック数
TESTRST_DIR = './test_rst/re_training/resnet34'#アーキテクチャごとに変更


#輪郭のみのデータセット あとで直す
train_path = "dataset/re_training/train_aug_img"
valid_path = "dataset/re_training/valid_img"
train_file_list = glob.glob(train_path+"/*.jpg")
valid_file_list = glob.glob(valid_path+"/*.jpg")
train_label = n2p(train_path,"label.csv")
valid_label = n2p(valid_path,"label.csv")

test_num = "try1"
test_path ="dataset/re_training/test_img" 
# ネットワークの設定
model_ft = models.resnet34(weights = ResNet34_Weights.DEFAULT)#resnet50
#model_ft = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)#vit
#model_ft = models.vgg16(weights=models.VGG16_Weights.DEFAULT)#vgg16


# layers = []
# layers.extend(list(model_ft.children())[0:5])
# layers.extend(list(model_ft.children())[-2:-1])
# model_ft = nn.Sequential(*layers, nn.Flatten())

for param in model_ft.parameters():#勾配計算?
    param.requires_grad = True


#最終ノードの出力を2に変更する
model_ft.fc = nn.Linear(model_ft.fc.in_features, 2)#resnet50で使用
#model_ft.heads[0] = nn.Linear(768, 2)#vitで使用
#model_ft.classifier[6] = nn.Linear(in_features=4096, out_features=2)#vgg16で使用
#model_ft.fc = nn.Linear(256, 2)#resnet50_layer1のみ


class ImageTransform(object):
    """
    入力画像の前処理クラス
    画像のサイズをリサイズする
    
    Attributes
    ----------
    resize: int
        リサイズ先の画像の大きさ
    mean: (R, G, B)
        各色チャンネルの平均値
    std: (R, G, B)
        各色チャンネルの標準偏差
    """
    def __init__(self, resize, mean, std):
        self.data_trasnform = {
            'train': transforms.Compose([
                # データオーグメンテーション
                transforms.RandomHorizontalFlip(),
                # 画像をresize×resizeの大きさに統一する
                transforms.Resize((resize, resize)),
                # Tensor型に変換する
                transforms.ToTensor(),
                # 色情報の標準化をする
                transforms.Normalize(mean, std)
            ]),
            'valid': transforms.Compose([
                # 画像をresize×resizeの大きさに統一する
                transforms.Resize((resize, resize)),
                # Tensor型に変換する
                transforms.ToTensor(),
                # 色情報の標準化をする
                transforms.Normalize(mean, std)
            ])
        }
    
    def __call__(self, img, phase='train'):
        return self.data_trasnform[phase](img)
    
class MyDataset(data.Dataset):
    """
    鳥のDataseクラス。
    PyTorchのDatasetクラスを継承させる。
    
    Attrbutes
    ---------
    file_list: list
        画像のファイルパスを格納したリスト
    classes: list
        鳥の特徴のラベル名
    transform: object
        前処理クラスのインスタンス
    phase: 'train' or 'valid'
        学習か検証かを設定
    """
    def __init__(self, file_list, label_path, classes, transform=None, phase='train'):
        self.file_list = file_list
        self.label_path = label_path
        self.transform = transform
        self.classes = classes
        self.phase = phase
    
    def __len__(self):
        """
        画像の枚数を返す
        """
        return len(self.file_list)
    
    def __getname__(self, index):
        return self.file_list[index]
    
    def __getitem__(self, index):
        """
        前処理した画像データのTensor形式のデータとラベルを取得
        """
        # 指定したindexの画像を読み込む
        img_path = self.file_list[index]
        img = Image.open(img_path).convert("RGB")
        #img = np.array(img)
        
        # 画像の前処理を実施
        img_transformed = self.transform(img, self.phase)
        
        # 画像ラベルをファイル名から抜き出す
        # ★ここの処理は各ファイルパスからフォルダ名(=クラス名)を抽出する処理です

        img_name = os.path.basename(img_path)
        df = pd.read_csv(self.label_path)
        label = df[df["name"]==img_name]["label"]
        
        # ラベル名を数値に変換
        # label = self.classes.index(str(int(label)))
        label = label.iloc[0]
        
        return img_transformed, label

#損失関数について
class Focal_MultiLabel_Loss(nn.Module):
    def __init__(self, gamma):
      super(Focal_MultiLabel_Loss, self).__init__()
      self.gamma = gamma
      #self.bceloss = nn.BCELoss(reduction='none')
      self.bceloss = nn.CrossEntropyLoss(reduction='none')


    def forward(self, outputs, targets): 
      bce = self.bceloss(outputs, targets)
      bce_exp = torch.exp(-bce)
      focal_loss = (1-bce_exp)**self.gamma * bce
      return focal_loss.mean()




# リサイズ先の画像サイズ
resize = 224

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

# Datasetの作成
train_dataset = MyDataset(
    file_list=train_file_list, label_path=train_label,
    classes=liver_classes,
    transform=ImageTransform(resize, mean, std),
    phase='train'
)
valid_dataset = MyDataset(
    file_list=valid_file_list, label_path=valid_label,
    classes=liver_classes,
    transform=ImageTransform(resize, mean, std),
    phase='valid'
)


# DataLoaderを作成
train_dataloader = data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)

valid_dataloader = data.DataLoader(
    valid_dataset, batch_size=1, shuffle=False)
#batch_sizeは一度のイテレーションで学習させるデータの量です
#shuffleはデータ利用時にランダムにデータを並べ替えるかどうか

# 辞書にまとめる
dataloaders_dict = {
    'train': train_dataloader, 
    'valid': valid_dataloader
}


# GPUの利用有無の設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)
net = model_ft.to(device)

# 損失関数
# criterion = nn.CrossEntropyLoss() #交差エントロピー
criterion = Focal_MultiLabel_Loss(gamma=4) #focal loss

# 最適化に関しては、いくつかのパターンを調べた結果、下記が一番結果がよかった
optimizer = optim.Adam(net.parameters(),lr=0.0001)

max_score = 0

if TRAIN:
    try_num = 1    
    # try_num = "try" + str(len(os.listdir(WEIGHTS_DIR)))
    # if not os.path.exists(os.path.join(WEIGHTS_DIR,try_num)):
    #     os.mkdir(os.path.join(WEIGHTS_DIR,try_num))

    for epoch in range(num_epochs):
        print('\nEpoch {}/{}'.format(epoch+1, num_epochs))
        print('-------------')
        
        for phase in ['train', 'valid']:
            
            if phase == 'train':
                # モデルを訓練モードに設定
                net.train()
            else:
                # モデルを推論モードに設定
                net.eval()
            
            # 損失和
            epoch_loss = 0.0
            # 正解数
            epoch_corrects = 0
            
            # DataLoaderからデータをバッチごとに取り出す
            for inputs, labels in tqdm(dataloaders_dict[phase]):
                # optimizerの初期化
                optimizer.zero_grad()
                
                # 学習時のみ勾配を計算させる設定にする
                with torch.set_grad_enabled(phase == 'train'):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = net(inputs)
                
                    #損失を計算
                    loss = criterion(outputs,labels)
                    
                    # ラベルを予測
                    _, preds = torch.max(outputs, 1)
                    
                    

                    # 訓練時はバックプロパゲーション
                    if phase == 'train':
                        # 逆伝搬の計算
                        loss.backward()
                        # パラメータの更新
                        optimizer.step()
                    
                    # イテレーション結果の計算
                    # lossの合計を更新
                    # PyTorchの仕様上各バッチ内での平均のlossが計算される。
                    # データ数を掛けることで平均から合計に変換をしている。
                    # 損失和は「全データの損失/データ数」で計算されるため、
                    # 平均のままだと損失和を求めることができないため。
                    epoch_loss += loss.item() * inputs.size(0)
                    
                    # 正解数の合計を更新
                    epoch_corrects += (torch.sum(preds == labels.data))/preds.size()[0]
            print(len(dataloaders_dict[phase]))
            # epochごとのlossと正解率を表示
            epoch_loss = epoch_loss / len(dataloaders_dict[phase])
            epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase])
            # モデルをepochごとにsave
            # if phase == 'valid':
                # if max_score < epoch_acc:
                #     print("Model save!")
                #     torch.save(net.state_dict(), 'model-'+str(epoch)+'_'+phase+'.pth')
                #     torch.save(net.state_dict(), os.path.join(WEIGHTS_DIR,try_num)+'/best_model.pth')
                #     max_score = epoch_acc
                
                # トレーニングの最後のエポックで重みを保存する
            if epoch == num_epochs - 1:
                print("Saving final model weights...")
                torch.save(net.state_dict(), os.path.join(WEIGHTS_DIR,"try1") + '/final_model.pth')
            print('{} Loss: {:.4f} Acc: {:.4f} Correct: {}/{}'.format(phase, epoch_loss, epoch_acc, epoch_corrects,len(dataloaders_dict[phase].dataset)))



if TEST:

    if not os.path.exists(os.path.join(TESTRST_DIR,test_num)):
        os.mkdir(os.path.join(TESTRST_DIR,test_num))
    
    # 統計情報を記録するCSVファイルのパス
    summary_csv_path = os.path.join(TESTRST_DIR, test_num, 'test_result_summary.csv')
    # 初期化
    correct凸あり = 0
    correct凸なし = 0
    total凸あり = 0
    total凸なし = 0
    
    # 学習済みモデルの読み込み
     
    model_ft.load_state_dict(torch.load(WEIGHTS_DIR+test_num+'/final_model.pth'))#tryを変える
    
    #---------------評価---------------#
    files = glob.glob(test_path+'/*.jpg')
    correct = 0
    test_pred = []#precisionのリスト
    test_true = []#testのリスト
    with open(os.path.join(TESTRST_DIR,test_num)+'/label.txt', "a") as f:
        for item in files:
            print(item)
            img = Image.open(item).convert("RGB")

            transform = ImageTransform(resize, mean, std)
            img_transformed = transform(img, 'valid').unsqueeze(0).to(device)#次元を追加。batch化の代わり
            
            pred = []
            with torch.no_grad():
                output = net(img_transformed)
                _, preds = torch.max(output, 1)
                pred += [int(l.argmax()) for l in output]
                print(pred)#[0]だと「可愛い」判定、[1]だと「かっこいい」判定です
                m = nn.Softmax(dim=1)
                print(m(output))#可愛いとかっこいい判定の度合いを表す評価値です
                
                img_name = os.path.basename(item)
                #df = pd.read_csv('./Dataset/training/test_ds/label.csv')#肝臓全体
                df = pd.read_csv(n2p(test_path,'label.csv'),encoding="shift-jis")#輪郭のみ
                label = df[df["name"]==img_name]["label"]
                label = label.iloc[0]

                
                test_pred.extend(pred)
                test_true.append(label)
                # 凹凸あり or 凹凸なしの分類
                if label == 1:  # 凹凸あり
                    total凸あり += 1
                    if label == preds.cpu().numpy():
                        correct凸あり += 1
                else:  # 凹凸なし
                    total凸なし += 1
                    if label == preds.cpu().numpy():
                        correct凸なし += 1
                if label == preds.cpu().numpy():
                    correct += 1
                print(item,pred,m(output), file=f, sep="\n")
        print(' Precision: {:.4f} Recall: {:.4f} Numeber of Correct: {}/{}'.format(precision_score(test_true,test_pred), recall_score(test_true,test_pred), correct,len(files)), file=f)
    
    with open(os.path.join(TESTRST_DIR,test_num)+'/test_excel.csv','w',newline ="") as csv_file:
        test_name = ["",test_num]
        fieldnames = ["name","凹凸あり推測値"]
        writer =csv.writer(csv_file)
        writer.writerow(test_name)
        writer.writerow(fieldnames)
        for item in files:
            #print(item)
            img = Image.open(item).convert("RGB")

            transform = ImageTransform(resize, mean, std)
            img_transformed = transform(img, 'valid').unsqueeze(0).to(device)#次元を追加。batch化の代わり
            
            pred = []
            with torch.no_grad():
                output = net(img_transformed)
                _, preds = torch.max(output, 1)
                pred += [int(l.argmax()) for l in output]
                #print(pred)#[0]だと「可愛い」判定、[1]だと「かっこいい」判定です
                m = nn.Softmax(dim=1)
                #print(m(output))#可愛いとかっこいい判定の度合いを表す評価値です
                #print(m(output)[0,1].item())
                img_name = os.path.basename(item)
                df = pd.read_csv(n2p(test_path,'label.csv'),encoding="shift-jis")#輪郭のみ
                label = df[df["name"]==img_name]["label"]
                label = label.iloc[0]

                
                test_pred.extend(pred)
                test_true.append(label)
                if label == preds.cpu().numpy():
                    correct += 1
                
                writer.writerow([img_name,m(output)[0,1].item()])
                # print(item,pred,m(output), file=f, sep="\n")


    # 統計情報をCSVに記録
    with open(summary_csv_path, 'w', newline="") as summary_csv:
        fieldnames = [
            "Category", "Accuracy", "Correct_Count", "Total_Count"
        ]
        writer = csv.DictWriter(summary_csv, fieldnames=fieldnames)
        writer.writeheader()
        
        # 凹凸あり
        accuracy凸あり = correct凸あり / total凸あり if total凸あり > 0 else 0
        writer.writerow({
            "Category": "凸あり",
            "Accuracy": f"{accuracy凸あり:.4f}",
            "Correct_Count": correct凸あり,
            "Total_Count": total凸あり
        })
        
        # 凹凸なし
        accuracy凸なし = correct凸なし / total凸なし if total凸なし > 0 else 0
        writer.writerow({
            "Category": "凸なし",
            "Accuracy": f"{accuracy凸なし:.4f}",
            "Correct_Count": correct凸なし,
            "Total_Count": total凸なし
        })
        
        # 全体
        overall_accuracy = correct / len(files) if len(files) > 0 else 0
        writer.writerow({
            "Category": "全体",
            "Accuracy": f"{overall_accuracy:.4f}",
            "Correct_Count": correct,
            "Total_Count": len(files)
        })

    print(f"Test result summary saved to {summary_csv_path}")

        # print(test_pred)
        # print(test_true)
        # cm =confusion_matrix(test_true,test_pred)
        # sns.heatmap(cm, annot=True, cmap='Blues')
        # plt.savefig(R'C:\Users\itsuk\python_files\fatty_liver\liver_behind_judge\confusion_matrix\contour/sklearn_confusion_matrix_annot_blues.png')
        # print(' Precision: {:.4f} Recall: {:.4f} 正解数は: {}/{}'.format(precision_score(test_true,test_pred), recall_score(test_true,test_pred), correct,len(files)))
        

    
    
    


3
if GRAD:  
    model_ft.load_state_dict(torch.load(WEIGHTS_DIR+'/try5/best_model.pth'))#tryを変える
    #model_ft.load_state_dict(torch.load('./resnet50_best_model.pth')) #デモの時使用
    model_ft.eval()
    target_layer = model_ft.layer4

    # def backward_hook(module, inputs, output):
    #     print('input:', inputs[0].shape, 'output', output.shape)  

    # target_layer.register_backward_hook(backward_hook)



    gradcam = GradCAM(model_ft, target_layer)
    gradcam_pp = GradCAMpp(model_ft, target_layer)
    #folder = './Dataset/training/test_ds'#肝臓全体のテストセット
    folder = './dataset/nochange/test_ds'#輪郭のみのテストセット
    files = glob.glob(folder+'/*.jpg')

    images = []
    # あるラベルの検証用データセットを呼び出してる想定
    for path in files:
        basename = os.path.basename(path)
        img = Image.open(path).convert("RGB")
        # torch_img = transforms.Compose([
        #     transforms.Resize((224, 224)),
        #     transforms.ToTensor()
        # ])(img).to(device)
        # normed_torch_img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(torch_img)[None]

        transform = ImageTransform(resize, mean, std)
        torch_img = transform(img, 'valid').unsqueeze(0).to(device)
        # normed_torch_img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(torch_img)[None]

        mask, _ = gradcam(torch_img,class_idx=1)
        heatmap, result = visualize_cam(mask, torch_img)

        mask_pp, _ = gradcam_pp(torch_img)
        heatmap_pp, result_pp = visualize_cam(mask_pp, torch_img)

        # images.extend([torch_img.cpu(), heatmap, heatmap_pp, result, result_pp])

        # plt.imshow(heatmap_pp.permute(1,2,0).cpu().detach().numpy())
        # plt.show()

        heatmap = (heatmap_pp.permute(1,2,0)*255).byte().cpu().detach().numpy()
        input_image = torch_img.squeeze(0).permute(1,2,0).cpu().detach().numpy()
        input_image = np.array(img)
        input_image = cv2.resize(cv2.cvtColor(input_image,cv2.COLOR_RGB2BGR),[resize,resize])
        heatmap = cv2.cvtColor(heatmap,cv2.COLOR_RGB2BGR)

        out = cv2.addWeighted(input_image,0.5,heatmap,0.5,0)

        # cv2.imshow("image",out)
        cv2.imwrite(os.path.join('./gradcam_rst/nochange_alb',basename), out)#layerごとに名前の変更
    

    # grid_image = make_grid(images, nrow=1)

    # # 結果の表示
    # out = transforms.ToPILImage()(grid_image)
    # print(grid_image.shape)

if MODEL:
    print(model_ft)

if PRcurve:
    precision, recall, thresholds = metrics.precision_recall_curve(test_true, test_pred)

    auc = metrics.auc(recall, precision)
    print(auc)

    plt.plot(recall, precision, label='PR curve (area = %.2f)'%auc)
    plt.legend()
    plt.title('PR curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.show()


