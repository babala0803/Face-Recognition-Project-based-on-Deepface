# CS308-Project-Face-Recognition
## File structure
- `def find_all.py` includes the function improved from the original function find from the Package deepface. It needs to be added to Deepface.py.
- `demo.ipynb` includes evaluation for both verification and identification for VGGFace2's evaluation set. The result will be saved in folder *fig*.
- `additional_demo.ipynb` includes evaluation for both verification and identification for our additional evaluation set or yours. The database comes from our additional evaluation set or yours and VGGFace2. The result will be saved in folder *add_fig*.
- `get_testset.ipynb` includes the generation of evaluation set. The first two blocks are for `demo.ipynb` and the rest two are for `additional_demo.ipynb`.
- `identity.csv` includes the name of people corresponding to the folder name. This will be used in prediction. If you add your dataset, remember to add this information to this file.
- `vgg_show.ipynb` includes metrics and testing.
- `util.py` includes methods `vgg_show.ipynb`  needs.
## Reproducing the results
- Install dependencies
```
pip install deepface
conda install sklearn
conda install chardet
```
​	or you can simply import `env.yaml`.

- Prepare evaluation set

  For evaluating face identification, we use VGGFace2's evaluation set first. You need to get this dataset on your own and rename the folder as *"vggface2"*.
  The folder structure should be like `./vggface2/n000001/n001_01.jpg` and the names of duplicated images from VGGFace2 will be the original folder name + '_' + serial number To prepare this dataset, you need to run the first two blocks of `get_testset.ipynb`. Here the folders of different celebrities in VGGFace2 can be selected on your own. Two folders *testset1* and *testset2* will be generated.

  If you want to see the performance of our additional evaluation set, run the third and forth blocks of `get_testset.ipynb`. You can also add your images with folders, each for one person. Two folders *add_testset1* and *add_testset2* will be generated.

- Add function `find_all`

  add the content in  `def find_all.py` to `Deepface.py` of the package.

- See the predicted result images

  Build an empty folder *fig* and run `demo.ipynb` if you use VGGFace2's evaluation set.

  Build an empty folder *add_fig* and run `additional_demo.ipynb` if you use our additional evaluation set or yours.
  
- Check the metrics and testing

​		Run the `vgg_show.ipynb`
