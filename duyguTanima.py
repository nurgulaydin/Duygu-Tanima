import sys, os         
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, AveragePooling2D

from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import np_utils

veri_cercevesi = pd.read_csv("C:/Users/bilgisayarım/Desktop/duygu_Tanima/fer2013.csv")   # pandas, csv dosyasını bir veri çerçevesi olarak okur.
x_egitim,egitim_y = [],[]                                 # eğitim için  girdi(x)  ve çıktı(y). array türünde
x_test,test_y = [],[]                                   # test için girdi(x) ve çıktı(y). array türünde.

for index, satir in veri_cercevesi.iterrows():            # iterrows(), hem index hem de satır verir. index'e göre veri çerçevesinin tekrarlarında(satırlarında) gezecek.
    value = satir["pixels"].split(" ")                    # satırların pixels sütununu boşluklardan böl. Liste olarak tutar.
    try:                                                # hata ayıklama
        if "Training" in satir["Usage"]:                  # satırların Usage sütunundaki  değerleri Tranining ise:
            x_egitim.append(np.array(value,"float32"))   # eğitim girdilerine piksel değerlerini ata. eğitim girdileri np.array türünde.
            egitim_y.append(satir["emotion"])              # eğitim çıktılarına duygu değerini ata.
        elif "PublicTest" in satir["Usage"]:              
            x_test.append(np.array(value,"float32"))    # test girdilerine.
            test_y.append(satir["emotion"])               # test çıktılarına.
    except:                                             # hata varsa çalışır.
        print(" İndex:{} ve Satır:{} 'da hata oluştu ".format(index,satir))

duygu_etiketleri = 7             # 7 tane duygu,durumu var.
batch_boyutu = 64                # Veri seti batch değeri olarak belirlenen değere göre parçalara ayrılır. Ve her tekrarda modelin eğitimi bu parça üzerinden yapılmaktadır. Bununla birlikte bazı durumlarda veri kendi içinde gruplanmış olabilmektedir. 32, 64, 128, 256 olabilir. 
epochs = 100                     # eğitim sayısı.
genislik, yukseklik = 48, 48     # fotoğrafların ölçüleri.

x_egitim = np.array(x_egitim,"float32")             # array'dan np.array' e çevrildi.
egitim_y = np.array(egitim_y,"float32")             # Numpy dizileri python listelerine benzer fakat hız ve işlevsellik açısından python listelerinden daha kullanışlıdır.

x_test = np.array(x_test,"float32")
test_y = np.array(test_y,"float32")

                                                                                # Dönüştürme ve ön işleme
egitim_y = np_utils.to_categorical(egitim_y,num_classes=duygu_etiketleri)       # Çıkış katmanından 7 farklı rakam yerine 0 veya 1 değerinin elde edildiği durum.                                                                         
test_y = np_utils.to_categorical(test_y,num_classes=duygu_etiketleri)           # Bu sayede doğru rakama karşılık gelen etiketin indeksi 1 iken diğer tüm etiketler için bu indeks 0 değerini alır. Bu şekilde etiketlerimizi bir vektöre dönüştürmüş oluyoruz.
                                                                                # hem eğitim hem de test çıktısı normalde 1-7 arası duygu etiket değerlerini tutuyor. Ancak np_utils.to_categorical() ile hangi etikete karşılık gelirse onun değeri 1 diğerleri 0 olmuş oluyor.

x_egitim -= np.mean(x_egitim,axis=0)               # np.mean() verilen dizinin değerlerinin ortalamasını bulmak için kullanılır. eğitimden ortalamaları çıkarıldı.
x_egitim /= np.std(x_egitim,axis=0)                # np.std() standart sapmayı bulmak için kullanılır. Arka planda bu işlemleri yapar ve sonucu bize getirir. eğitim standart sapmaya bölündü.

x_test -= np.mean(x_test,axis=0)                   # axis hangi bazda işlem yapılacağını açıklar. eğer axis 0 olursa satır bazında axis 1 olursa sütun bazında işlem yapılır.  
x_test /= np.std(x_test,axis=0)

x_egitim = x_egitim.reshape(x_egitim.shape[0], 48, 48, 1)   # shape[0] satır sayısını, shape[1] sütun sayısını verir.
x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)         #  Reshape (yeniden şekillendirme), bir dizinin şeklini (shape) değiştirmek anlamına gelir.
                                                            # Bir dizinin şekli (shape), dizinin her boyutundaki (1D, 2D, 3D) elemanlarının sayısıdır.
                                                            # Burada eğitim ve test verileri 48 satır, 48 sütun ve  her sütunda 1 veri(1 resim değerleri) olacak şekilde 3 boyutlu hale dönüştürüldü.

model = Sequential()                                    # CNN modeli oluşturmak için. 

# ilk önce giriş katmanı yazılır.                       # Bilgisayarlı görü çalışmalarında Conv2d katmanı kullanılır.
model.add(Conv2D(64, kernel_size=(3,3), activation="relu", input_shape=(x_egitim.shape[1:])))
                                                                  # Oluşturulan ağda her biri 3x3 boyutunda olan 64 konvolüsyonel filtreyi(nöron) öğrenir. Relu fonk. kullanılır. Output'a relu uygulanır.
                                                                  # input olarak eğitimin şekli kullanıldı.(sütun bazında işlem yapacak. Eğitim verileri 48 satır, 48 sütun ve  her sütunda 1 veri olacak şekilde 3 boyutlu hale dönüştürüldüğü için sütunlar bitene kadar işlem yapılacak.)
model.add(Conv2D(64, kernel_size=(3,3), activation="relu"))       # 1. gizli katman. Nöron sayısı isteğe bağlı ancak (giriş+çıkış/2) mantıklıdır. Küçük veri seti için çok fazla gizli katman mantıklı olmaz.

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
                                                           # MaxPooling2D, görüntülerde aşırı öğrenmeyi engellemek ve öğrenme süresini kısaltmak için kullanılır.Feature map'te Max değeleri seçer.
                                                           # pool_size, max değeleri seçerken input(48x48) üzerine (2x2) boyutlu matris ile max değeleri alacak. 
                                                           # Yani 2 satır 2 sütundan 4 karedeki piksel değerlerinden en büyüğünü alacak. 48x48 içindeki her dörtlüde bunu yapacak.  
                                                           # strides, ile de 2x2 birim kayacak. 4 kareden sonra hareket ederek yandaki 4 kareye gelecek. 
model.add(Dropout(0.5))                         # Eğitim sırasında aşırı öğrenmeyi engellemek için bazı nöronları unutmak için kullanılanılır.  
                                                # Dropout’ 0.5 ten başlayarak model maksimum performansa ulaşıncaya kadar sayı azaltılabilir.
model.add(Conv2D(64, (3,3), activation="relu"))          # 2. gizli katman.
model.add(Conv2D(64, (3,3), activation="relu"))          # 3. gizli katman.
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.5))

model.add(Conv2D(128, (3,3), activation="relu"))         # 4.gizli katman.
model.add(Conv2D(128, (3,3), activation="relu"))         # 5.gizli katman.
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Flatten())                                     # Flatten(), matris formundaki veriyi düzleştirir. Sinir ağı tek boyutlu dizi kullanır.

                                                         # 6.gizli katman.
model.add(Dense(1024, activation="relu" ))               # Katmanları temsil eden “Dense” en çok kullanılan sınıflardan biridir. 
model.add(Dropout(0.2))                                  # Bir katmandan aldığı nöronların bir sonraki katmana girdi olarak bağlanmasını sağlar.
model.add(Dense(1024, activation="relu"))                # 7.gizli katman.
model.add(Dropout(0.2))

                                                                                           # Çıktı katmanı. Son katman.
model.add(Dense(duygu_etiketleri, activation="softmax"))                                   # Softmax Fonksiyonu: verilen her bir girdinin bir sınıfa ait olma olasılığını gösteren [0,1] arası çıktılar üretmektedir. Çoklu sınıflandırma probleminde kullanılır.                                         
model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['accuracy'])       # compile () fonksiyonu, model hazır olduğunda, öğrenme sürecini yapılandırır.
                                                                                           # categorical_crossentropy, Çok sınıflı sınıflandırma görevlerinde kullanılan bir kayıp fonksiyonudur(hedefin gerçek ve tahmin edilen değeri arasındaki mesafeyi ölçer.)
                                                                                           # Örneğin resimdeki bir meyvenin elma, armut veya muz olacak şekilde sınıflandırılması. Genellikle softmax aktivasyon fonksiyonunun ardından kullanılır. Bu yüzden softmax loss olarak da anılır.
                                                                                           # Optimize edici(uygun hale getirmek) olarak “adam” kullanılır. Adam algoritması, eğitim boyunca öğrenme oranını ayarlar. Öğrenme oranı, model için uygun ağırlıkların ne kadar hızlı hesaplandığını belirler.
                                                                                           # metrics=['accuracy'] , eğitim ve test sırasında değerlendirmek istenen metrikler yazılır. Örn. burada accuracy(doğruluk) oranı isteniyor.

model.fit(x_egitim, egitim_y, batch_size=batch_boyutu, epochs=epochs, 
                    verbose=1, validation_data=(x_test, test_y), shuffle=True )    # Model yapılandırıldıktan sonra eğitim süreci model.fit () ile yapılır.                                                         
                                                                                  # Eğitim verileri, batch büyüklüğünü ve epoch(eğitim sayısı) verilir.
                                                                                  # Verbose, eğitim ilerlemesini nasıl 'görmek' istediğinizi ayarlar. verbose=0 hiçbir şey göstermez(sessiz). verbose=1 animasyonlu ilerleme çubuğu gösterir [=======]. verbose=2 epoch sayısını gösterir Epoch 1/10.
                                                                                  # Validation_data, doğrulama(test) verileri. shuffle=True, eğitim verilerinin sırasını her epoch'dan önce rastgele değiştirir.
fer_json = model.to_json()     # JSON olarak kaydetme. Yalnızca bir modelin mimarisini kaydeder. Oluşturulan JSON / YAML dosyaları insan tarafından okunabilir ve gerektiğinde manuel olarak düzenlenebilir.

with open("fer.json", "w") as json_file:   # json dosyası oluşturma with open() ile olur. fer.json, oluşturulan dosya. "w" ise o dosyaya yazma işlemi olacak demek. 
    json_file.write(fer_json)              # json_file takma ad gibi düşünülebilir. Yukarıda json'a çevirip kaydettiğimiz modeli Burada oluşturduğumuz json'a yazacağız.

model.save_weights("fer.h5")               # model.save_weights(): Yalnızca ağırlıkları kaydeder, böylece ihtiyacınız olursa bunları farklı bir mimariye uygulayabilirsiniz. mode.save(): Modelin mimarisini + ağırlıkları + eğitim konfigürasyonunu + optimize edicinin durumunu kaydeder.
                                           # Yalnızca adını verirseniz model.save_weights("fer"), kaydedilen model biçimi varsayılan olarak TF olacaktır.H5 dosya uzantısını sağlarsanız model.save_weights("fer.h5"), kaydedilen model formatı HDF5 olacaktır.
                                           # Hiyerarşik Veri Formatı (HDF) Dosyası mühendislik, tıp, fizik, havacılık, finans ve akademik araştırma gibi birçok alanda kullanılmaktadır. Bilimsel verilerin çok boyutlu dizilerini içerir(data files).
                                           # TIF dosyası özünde bir görüntü dosyasıdır, her türlü yüksek çözünürlükteki görüntüleri kapsar.
