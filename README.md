# Face recognition cnn
Face verification and recognition using a pre-trained <a href="https://arxiv.org/pdf/1503.03832.pdf">FaceNet</a> deep convolutional neural network:

<p align="center"><img src="https://user-images.githubusercontent.com/24521991/33219663-8e0487de-d17e-11e7-86ff-0312ef3970a8.png" width="700"></p>

<br/>

### 1. Database
The database are my classmates from Tsinghua University:

<p align="center"><img src="https://user-images.githubusercontent.com/24521991/33219850-b0b2a51c-d17f-11e7-8ebb-e938e3272bf4.PNG" width="500"></p>

If you want to have your own database just need to add your own face images (96x96) to the <b>/images</b> folder. Then modify below lines of code accordingly:

```python
# fr.py
44. database = {}
45. database["yimin"] = img_to_encoding("images/yimin.jpg", FRmodel)
46. database["alex"] = img_to_encoding("images/alex.jpg", FRmodel)
47. database["white"] = img_to_encoding("images/white.jpg", FRmodel)
48. database["jiayi"] = img_to_encoding("images/jiayi.jpg", FRmodel)
49. database["kevinthu"] = img_to_encoding("images/kevinthu.jpg", FRmodel)
50. database["jane"] = img_to_encoding("images/jane.jpg", FRmodel)
51. database["lucky"] = img_to_encoding("images/lucky.jpg", FRmodel)
52. database["bruno"] = img_to_encoding("images/bruno.jpg", FRmodel)
53. database["adeline"] = img_to_encoding("images/adeline.jpg", FRmodel)
54. database["sdt"] = img_to_encoding("images/sdt.jpg", FRmodel)
55. database["alvaro"] = img_to_encoding("images/alvaro.jpg", FRmodel)
56. database["linda"] = img_to_encoding("images/linda.jpg", FRmodel)
```

### 2. Face Verification
Face verification verifies the input face ("alvaro_0.jpg") encoding vector corresponds (distance < threashold) to the provided name ("alvaro") database member encoding vector: 

```python
# fr.py
76. verify("images/alvaro_0.jpg", "alvaro", database, FRmodel)
```

<p align="center"><img src="https://user-images.githubusercontent.com/24521991/33221585-908bef08-d18b-11e7-903b-330a73195b64.png" width="700"></p>

### 3. Face Recognition
Face recognition compares the input face ("alvaro_0.jpg") encoding vector with all database members encoding vector, chosing the one with the minimum distance between vectors:

```python
# fr.py
106. who_is_it("images/alvaro_0.jpg", database, FRmodel)
```

<p align="center"><img src="https://user-images.githubusercontent.com/24521991/33221634-f4b4fb3c-d18b-11e7-9604-a9e488d98ed6.png" width="800"></p>

### 4. Final output

<p align="center"><img src="https://user-images.githubusercontent.com/24521991/33221651-0cc5f44c-d18c-11e7-921b-ec5710aa2b7a.png" width="500"></p>
