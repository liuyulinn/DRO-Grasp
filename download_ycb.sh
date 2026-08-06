pip install gdown

### xhand right 
gdown https://drive.google.com/uc?id=1pl5jJE1tzTvqkt8gzfixn6gp5Cy5HCO_
mkdir -p ./data/bodex

tar -xf ycb_xhand_right_lifted.tar.gz -C ./data/bodex
rm ycb_xhand_right_lifted.tar.gz

### xhand left 
gdown https://drive.google.com/uc?id=1cZSklqTUHOqX0lCgZc7wsH0g59Ev0RUx
mkdir -p ./data/bodex

tar -xf ycb_xhand_left_lifted.tar.gz -C ./data/bodex
rm ycb_xhand_left_lifted.tar.gz

### fixsharpa_right 
gdown https://drive.google.com/uc?id=1VChaskI78yNUWFCS_1xsPs4Ty6lkdMB4
mkdir -p ./data/bodex

tar -xf ycb_fixsharpa_right_lifted.tar.gz -C ./data/bodex
rm ycb_fixsharpa_right_lifted.tar.gz

### fixsharpa_left
gdown https://drive.google.com/uc?id=1VbEhxY-XWDYcqL7EUVQSMsShx4UFluZY
mkdir -p ./data/bodex

tar -xf ycb_fixsharpa_left_lifted.tar.gz -C ./data/bodex
rm ycb_fixsharpa_left_lifted.tar.gz


### allegro_right 
gdown https://drive.google.com/uc?id=1yTj3Pq5FMmkGrZpCOQxedW55fqjaru7u
mkdir -p ./data/bodex

tar -xf ycb_allegro_right_lifted.tar.gz -C ./data/bodex
rm ycb_allegro_right_lifted.tar.gz

### allegro_left
gdown https://drive.google.com/uc?id=1Ye6P-Cw7UrEr2j9qJ3k-CGnXXKWVh1M-
mkdir -p ./data/bodex

tar -xf ycb_allegro_left_lifted.tar.gz -C ./data/bodex
rm ycb_allegro_left_lifted.tar.gz


### ycb objects
gdown https://drive.google.com/uc?id=1a76FuP10DaSq2AtrAWhSRTyl5-kT9faM
mkdir -p ./data/object

tar -xf ycb_collected.tar.gz -C ./data/object
rm ycb_collected.tar.gz
