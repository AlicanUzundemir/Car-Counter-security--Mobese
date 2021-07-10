import cv2
import numpy as np #kuracagımız matrisler icin kullanılır


Video_Okuyucu = cv2.VideoCapture("traffic.mp4") #video yu okuyoruz
fgbg =cv2.createBackgroundSubtractorMOG2()

kernel = np.ones((5,5),np.uint8) #matrisin oldusturgu tüm degerleri 1 yapıyor.

class Koordinat:
    def __init__(self,x,y):
        self.x = x
        self.y = y

class Sensor:
    def __init__(self,Koordinat1,Koordinat2,Kare_genislik,Kare_uzunluk):
        self.Koordinat1 = Koordinat1
        self.Koordinat2 = Koordinat2
        self.Kare_genislik = Kare_genislik
        self.Kare_uzunluk = Kare_uzunluk
        self.Maskenin_Alani =abs(self.Koordinat2.x-Koordinat1.x)*abs(self.Koordinat2.y-self.Koordinat1.y)#abs mutlak sayı
        self.maske = np.zeros((Kare_uzunluk,Kare_genislik,1),np.uint8)
        cv2.rectangle(self.maske,(self.Koordinat1.x,self.Koordinat1.y),(self.Koordinat2.x,self.Koordinat2.y),(255),thickness=cv2.FILLED)
        self.durum = False
        self.AlgılananAracSayisi = 0

Sensor1 = Sensor(Koordinat(200,300),Koordinat(240,250),640,360)
cv2.imshow("Maske",Sensor1.maske)

font = cv2.FONT_HERSHEY_SIMPLEX

while (1):
    ret, Kare =Video_Okuyucu.read()
    #print(Kare.shape)
    Kesilmis_Kare = Kare[0:360,0:640]
    #print(Kare.shape)

    Arka_Plan_Silinmis_Kare = fgbg.apply(Kare)

    Arka_Plan_Silinmis_Kare = cv2.morphologyEx(Arka_Plan_Silinmis_Kare,cv2.MORPH_OPEN,kernel)

    cnts,_=cv2.findContours(Arka_Plan_Silinmis_Kare,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    Sonuc = Kesilmis_Kare.copy()

    doldurulmusResim=np.zeros((Kesilmis_Kare.shape[0],Kesilmis_Kare.shape[1],1),np.uint8)#siyaharla dolu
    ret, maske = cv2.threshold(doldurulmusResim, 10, 255, cv2.THRESH_BINARY)
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        if (w>50 and h>30): #yesil kare icindeki kücük kücük olusan kareleri yok ettik
            cv2.rectangle(Sonuc,(x,y),(x+w,y+h),(0,255,0),thickness=2)
            cv2.rectangle(doldurulmusResim,(x,y),(x+w,y+h),(255),thickness=cv2.FILLED)

    #cv2.rectangle(Sonuc,(Sensor1.Koordinat1.x,Sensor1.Koordinat1.y),(Sensor1.Koordinat2.x,Sensor1.Koordinat2.y),(0,0,255),thickness=cv2.FILLED)
    
    Sensor1_Maske_Sonuc=cv2.bitwise_and(doldurulmusResim,doldurulmusResim,mask=Sensor1.maske)
    Sensor1_Beyaz_Piksel_Sayisi = np.sum(Sensor1_Maske_Sonuc==255)
    Sensor1_Oran = Sensor1_Beyaz_Piksel_Sayisi/Sensor1.Maskenin_Alani
    print(Sensor1_Oran) # maske de ekran sensor dısında cıkan yesil kare koordinaylarını gördük

    if(Sensor1_Oran>=0.75 and Sensor1.durum == False):
        cv2.rectangle(Sonuc,(Sensor1.Koordinat1.x,Sensor1.Koordinat1.y),(Sensor1.Koordinat2.x,Sensor1.Koordinat2.y),(0,255,0),thickness=cv2.FILLED)

        Sensor1.durum=True
    elif(Sensor1_Oran<=0.75 and Sensor1.durum == True):
        cv2.rectangle(Sonuc,(Sensor1.Koordinat1.x,Sensor1.Koordinat1.y),(Sensor1.Koordinat2.x,Sensor1.Koordinat2.y),(0,0,255),thickness=cv2.FILLED)

        Sensor1.durum=False
        Sensor1.AlgılananAracSayisi+=1
    else:
        cv2.rectangle(Sonuc,(Sensor1.Koordinat1.x,Sensor1.Koordinat1.y),(Sensor1.Koordinat2.x,Sensor1.Koordinat2.y),(0,0,255),thickness=cv2.FILLED)

    cv2.putText(Sonuc,str(Sensor1.AlgılananAracSayisi),(Sensor1.Koordinat1.x,Sensor1.Koordinat1.y),font,1,(255,255,255) )

    #else:
    #    cv2.rectangle(Sonuc,(Sensor1.Koordinat1.x,Sensor1.Koordinat1.y),(Sensor1.Koordinat2.x,Sensor1.Koordinat2.y),(0,0,255),thickness=cv2.FILLED)


    #cv2.imshow("Kare",Kare)
    #cv2.imshow("Arka Plan Silinmiş Kare",Arka_Plan_Silinmis_Kare)
    cv2.imshow("Doldurulmus Resim",doldurulmusResim)
    cv2.imshow("Sensor1 Maske Sonuc",Sensor1_Maske_Sonuc)
    cv2.imshow("Sonuc",Sonuc)

    k = cv2.waitKey(30) & 0xff
    if k==27:
        break

cv2.waitKey(0)
cv2.destroyAllWindows()


