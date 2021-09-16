# -*- coding: utf-8 -*-
front_cam = 0
side_cam = 1
full_speed = 20
turn_speed = full_speed * 0.8
EMLARGE_RATIO = 1.2
model_prefix="/home/root/autostart/src/"
# mession config
# one more for background
MISSION_NUM = 8
mission_low = 0.3
mission_high = 0.75
MISS_DURATION = 200
mission_label_list = {
	0: "background",
	1: "daijun",
	2: "dunhuang",
	3: "dingxiangjun",
	4: "target",
	5: "trophies"
}

# sign config
MAX_SIGN_PER_FRAME = 1
sign_list = {
	0: "background",
	1: "fortress",
	2: "target",
	3: "camp",
	4: "juxushan",
	5: "barracks",
  6: "end",
  7: "en"
}
# cruise model
cruise = {
	"model":model_prefix + "models/cruise"
}
# sign models
sign = {
	"model": model_prefix + "models/sign",
	"threshold": 0.35,
	"label_list": sign_list,
	# label = 0 is background
	"class_num": 10
}
# task model
task = {
	"model":model_prefix + "models/task",
	"threshold":0.6,
	"label_list":mission_label_list
}
