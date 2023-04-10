from image_to_music_module import *
import glob

file_path = os.path.join(os.get.cwd())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_together = torchvision.models.vgg19_bn(pretrained=True)
num_ftrs = model_together.classifier[6].in_features
model_together.classifier[6] = nn.Sequential(nn.Linear(num_ftrs, 500),
                                             nn.Linaer(500, 2))
model_together = model_together.to(device)

optimizer_together = optim.SGD(model_together.parameters(), lr=0.001, momentum=0.9)

model, optimizer = load(os.path.join(file_path, 'results/model_vgg_19_50_0.000500.pth'), model_together, optimizer_together)



data_transforms = {'train': transforms.Compose( [transforms.RandomResizedCrop(224), # 224로 바꿔준다.
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])] ), # rgb임으로 3개씩; 대중적으로 널리 알려진 값
                   'val': transforms.Compose( [transforms.Resize(256),
                                               transforms.CenterCrop(224), # 최종적으로 train과 같이 224로 바꿔준다.
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])] ) # 위와 동일
                   }

test_image_lst = glob(os.path.join(file_path, 'DATA', 'OASIS', 'test_real_5/*', ))

result_dir = os.path.join(file_path, 'modeling')

data_5 = test_data(test_image_lst, data_transforms['val'], result_dir, 'test_image_output_5.pt')

image_to_music(data_5, 'song_with_minmax.csv',5)
