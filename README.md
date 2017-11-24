# Face recognition cnn
Face verification and recognition using a pre-trained <a href="https://arxiv.org/pdf/1503.03832.pdf">FaceNet</a> deep convolutional neural network:

<p align="center"><img src="https://user-images.githubusercontent.com/24521991/33219663-8e0487de-d17e-11e7-86ff-0312ef3970a8.png" width="700"></p>

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

### 2. Verification
Verification is executed by the function <b>verify()</b>, this function compares the input image (in our case "alvaro_0.jpg") witn the database member with the name you provide (in our case "alvaro"):

```python
76. verify("images/alvaro_0.jpg", "alvaro", database, FRmodel)
```

<p align="center"><img src="https://user-images.githubusercontent.com/24521991/33220117-e2442cda-d180-11e7-8546-30327ebff1a6.png" width="700"></p>

### 3. Recognition
Recognition is executed by the function <b>who_is_it()</b>, this function compares the input image (in our case "alvaro_0.jpg") with the whole database choosing the member with the minimum distance:

```python
106. who_is_it("images/alvaro_0.jpg", database, FRmodel)
```

<p align="center"><img src="https://user-images.githubusercontent.com/24521991/33220185-477b33b4-d181-11e7-9752-cbd6e6792aef.png" width="800"></p>

### 4. Final output

<p align="center"><img src="https://user-images.githubusercontent.com/24521991/33220524-f4419e1a-d183-11e7-8f0f-b89f06f0d383.png" width="500"></p>
