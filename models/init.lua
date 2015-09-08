require 'recurrent/' -- this means you can only import the models folder (i.e, "require 'models/'" in train.lua) from the base directory of this repo (else recurrent/ won't exist). there must be a better way to do this, but I don't know what it is.
models = {}
paths.dofile('basic.lua')
return models
