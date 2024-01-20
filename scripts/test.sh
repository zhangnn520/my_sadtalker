# ### some test command before commit.
# python inference2.py --preprocess crop --size 256
# python inference2.py --preprocess crop --size 512

# python inference2.py --preprocess extcrop --size 256
# python inference2.py --preprocess extcrop --size 512

# python inference2.py --preprocess resize --size 256
# python inference2.py --preprocess resize --size 512

# python inference2.py --preprocess full --size 256
# python inference2.py --preprocess full --size 512

# python inference2.py --preprocess extfull --size 256
# python inference2.py --preprocess extfull --size 512

python inference.py --preprocess full --size 256 --enhancer gfpgan
python inference.py --preprocess full --size 512 --enhancer gfpgan

python inference.py --preprocess full --size 256 --enhancer gfpgan --still
python inference.py --preprocess full --size 512 --enhancer gfpgan --still
