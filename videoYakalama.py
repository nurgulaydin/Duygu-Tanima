import os            # dosya ve klasör yapılarıyla çalışmak için gerekli kütüphane
import cv2           # yüz tanıma resim alma işlemleri için gerekli kütüphane (görüntü işlemleri)
import numpy as np   # matris ve sayı dizileri için gerekli kütüphane

from tensorflow.keras.models import model_from_json            # derin öğrenme için kullanılan kütüphane. model_from_json, Bir JSON yapılandırma dosyası ayrıştırır ve bir model örneği döndürür.
from tensorflow.keras.preprocessing import image               # keras.preprocessing, veri ön işleme ve veri arttırma kütüphanesi. image, görüntüleri biçimlendirmek için kullanılır.

                                                       # modeli yükle
model = model_from_json(open("fer.json","r").read())   # fer.json dosyasını okuma için aç. model_from_json ile json modelini yükle.
model.load_weights("fer.h5")                           # ağırlıkları yükle 

                                                        # HaarCascade sınıflandırıcısı, bir resim ya da video karesi içerisinde bulunan belirli bir nesnenin tespit edilmesi amacıyla kullanılmaktadır.
                                                        # Belirtilen yoldaki HaarCascade xml dosyasını aç ve değişkene ata.
yuz_haar_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")       # yüz tespiti için kullanılacak olan XML dosya yolu. CascadeClassifier() ile bu xml okunur.

                                                # Videoyu yakala ve değişkene ata 
video_yakala = cv2.VideoCapture(0)              # VideoCapture(),kamera açmaya yarar. Harici bir kamerada 0 yerine 1,2,3..vs kullanılabilir.

                                                 # Döngü kur
while True:
                                                 # Videodan resmi yakala
    ret,resim=video_yakala.read()                # kamerayı read() ile okur.
                                                 # ret ve resim olarak 2 değişken olmasının nedeni: kamera bir değer tuttuğu zaman resim'in içine atıyor. 
                                                 # İkinci değeri tuttuğu zaman kaybolabiliyor. Kayıp olmaması için 2 değer daha mantıklı.
    if not ret:
        continue
                                                              # Resmi BGR renk formatına dönüştür ve değişkene ata  
                                                              # cvtColor() Görüntüyü bir renk uzayından diğerine dönüştürür.           
    gri_resim = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)       # COLOR_BGR2GRAY gri tonlamalı renge dönüştürme.
  
                                                                              #  HaarCascade xml dosyasında eşleşme bul ve değişkene ata 
    yuz_tanima = yuz_haar_cascade.detectMultiScale(gri_resim, 1.32, 5)        # detectMultiScale() algılama,eşleşme için kullanılır. Tanıma oranı(hassaslık değeri) 1.32 ve 5 arası. Tanıma olmadığı zaman hassaslık değerleri ayarlanır.
                                                                              # detectMultiScale, görüntülerde farklı boyutlardaki nesneleri algılar ve yüzlere yerleştirilmiş dikdörtgenleri döndürür.
                                                                              # İlk parametre görüntü, ikinci parametre ölçek faktörü (Her görüntü ölçeğinde görüntü boyutunun ne kadar küçültüleceğini belirten parametre). 
                                                                              # Üçüncü parametre her aday dikdörtgenin onu tutmak için kaç komşusu olması gerektiğini belirten parametre.Bu parametre, algılanan yüzlerin kalitesini etkiler: daha yüksek değer, daha az algılamayla, ancak daha yüksek kaliteyle sonuçlanır.
                                            
                                                   # x,y ve width (genişlik),height (yükseklik) koordinatları.
    for(x,y,w,h) in yuz_tanima:                    # x,y,w,h çizgileri oluşturulacak ve bunlar birleştirilecek. Dikdörtgen yapacak.
                                                   # Dikdörtgen oluştur (Belirlenen renk ve kalınlıkta). Yüzlerin tanındığını göstermek için. Yukarda tanıyor burada ekrana basıyor.
        cv2.rectangle(resim,(x,y),(x+w,y+h),
                            (255,255,0),thickness=3)        # rectangle(img,başlangıç,bitiş,renk,kalınlık), dikdörtgen demek.                    
                                                              # img:görüntü resim. Başlangıç:( X koordinat değeri, Y koordinat değeri). Bitiş: rectangle öğesinin bitiş koordinatlarıdır ( X+W koordinat değeri, Y+H koordinat değeri).  
                                                              # Renk: çizilecek rectangle sınır çizgisinin rengidir. örneğin: (255, 0, 0) mavi renk için. Kalınlık: çizilen şeklin çerçeve kalınlığı.
        yuz = gri_resim[y:y+w,x:x+h]                          # yüzü kes.
        yuz = cv2.resize(yuz, (48,48))                        #  resize = yeniden boyutlandırmak. Yüzü tekrar boyutlandır.              

        resim_piksel = image.img_to_array(yuz)                # Pixeller üzerinde çalış ve tahmin oluştur. img_to_array(), bir görüntü örneğini Numpy dizisine dönüştürür.
        resim_piksel = np.expand_dims(resim_piksel, axis=0)   # np.expand_dims(), belirtilen konuma yeni bir eksen ekleyerek diziyi genişletir. axis = 0 ile satır bazında işlem yapılır.
        resim_piksel /= 255

        tahminler = model.predict(resim_piksel)               # model.predict(), modele belirtilen girdi örneklerine bağlı olarak çıktı tahminlerinde bulunur. Duygu durum analizi.
        max_index = np.argmax(tahminler[0])                   # np.argmax(), belirli bir eksendeki dizi nin max öğesini getirir. İndex tutar.

        duygular = ["KIZGIN", "IGRENME", "KORKMUS",
                "MUTLU", "UZGUN", "SASKIN", "NORMAL" ]        # ifadeleri tanımla.
        tahmini_duygu = duygular[max_index]                  # ifadeyi tahmin edilen ifade değişkenine ata.

        cv2.putText(resim, tahmini_duygu, (int(x),int(y-10)), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255), 2  )  # Tahmin edilen ifadeyi yaz. cv2.putText()yöntemi, herhangi bir görüntüye bir metin dizesi çizmek için kullanılır.
                                                                                                           # image = Üzerine yazının çizileceği resim.
                                                                                                           # metin = Çizilecek metin dizisi. tahmini duygu. 
                                                                                                           # org: Resimdeki metin dizesinin sol alt köşesinin koordinatlarıdır.( X koordinat değeri, Y koordinat değeri).
                                                                                                           # font: Yazı tipini belirtir. Yazı tiplerinden bazıları FONT_HERSHEY_SIMPLEX, FONT_HERSHEY_PLAIN, vb.dir .
                                                                                                           # fontScale: Yazı tipine özgü taban boyutu ile çarpılan yazı tipi ölçek faktörü.
                                                                                                           # color: Çizilecek olan metin dizisinin rengidir.örneğin: (255, 0, 0) mavi renk için.
                                                                                                           # kalınlık: px cinsinden çizginin kalınlığıdır .
    yeniBoyut_resim = cv2.resize(resim, (1000, 600))               # Resmin tekrar boyutlandır ve değişkene ata. Pencerenin açılacağı boyut.
    cv2.imshow("Analiz Sonucu", yeniBoyut_resim)                   # Boyutlandırılan resim pencere başlığıyla ekranda gösterilir. imshow(), resimi ekranda gösterir.

    if cv2.waitKey(10) == ord('q'):                                # Çıkmak için q tuşuna bas ve döngü kırılır.
        break                                                      # waitkey() işlevi , kullanıcıların belirli bir milisaniye boyunca veya herhangi bir tuşa basılana kadar bir pencere görüntülemesine olanak tanır.
                                                                   # ord(), belirtilen bir karakterin unicode'unu temsil eden sayıyı döndürür.
                                                 
video_yakala.release()                                   # release(), kamera kapatılır.
cv2.destroyAllWindows()                                  # programı durdurur.