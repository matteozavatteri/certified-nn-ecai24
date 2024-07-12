NN_Inputs = ["mode", 
			"number_of_people", 
			"number_of_people_in_group", 
			"group_radius", 
			"distance_to_group", 
			"robot_within_group", 
			"robot_facing_group", 
			"robot_work_radius", 
			"distance_to_closest_human", 
			"distance_to_2nd_closest_human", 
			"distance_to_3rd_closest_human", 
			"direction_to_closest_human", 
			"direction_to_2nd_closest_human", 
			"direction_to_3rd_closest_human", 
			"direction_from_closest_human_to_robot", 
			"robot_facing_closest_human", 
			"robot_facing_2nd_closest_human", 
			"robot_facing_3rd_closest_human", 
			"closest_human_facing_robot", 
			"2nd_closest_human_facing_robot", 
			"3d_closest_human_facing_robot", 
			"number_of_children", 
			"distance_to_closest_child", 
			"number_of_animals", 
			"distance_to_closest_animal", 
			"number_of_people_on_sofa", 
			"music_playing", 
			"number_of_agents_in_scene"]

from collections import OrderedDict


FUB = OrderedDict()
FUB["mode"] = 1
FUB["number_of_people"] = 9
FUB["number_of_people_in_group"] = 5
FUB["group_radius"] = 1 # 50
FUB["distance_to_group"] = 6 # 50
FUB["robot_within_group"] = 1
FUB["robot_facing_group"] = 1
FUB["robot_work_radius"] = 3 # 2.987226
FUB["distance_to_closest_human"] = 5 # 50
FUB["distance_to_2nd_closest_human"] = 5 # 50
FUB["distance_to_3rd_closest_human"] = 5 # 50
FUB["direction_to_closest_human"] = 360 # 1000
FUB["direction_to_2nd_closest_human"] = 360 # 1000
FUB["direction_to_3rd_closest_human"] = 360 # 1000
FUB["direction_from_closest_human_to_robot"] = 360 # 1000
FUB["robot_facing_closest_human"] = 1
FUB["robot_facing_2nd_closest_human"] = 1
FUB["robot_facing_3rd_closest_human"] = 1
FUB["closest_human_facing_robot"] = 1
FUB["2nd_closest_human_facing_robot"] = 1
FUB["3d_closest_human_facing_robot"] = 1
FUB["number_of_children"] = 2
FUB["distance_to_closest_child"] = 6 # 50
FUB["number_of_animals"] = 1
FUB["distance_to_closest_animal"] = 6 # 50
FUB["number_of_people_on_sofa"] = 2
FUB["music_playing"] = 1
FUB["number_of_agents_in_scene"] = 11


NN_Outputs = ["vacuum_cleaning", 
			"mopping_the_floor", 
			"carry_warm_food", 
			"carry_cold_food", 
			"carry_drinks", 
			"carry_small_objects", 
			"carry_big_objects", 
			"cleaning_or_starting_conversation"]

Fieldnames = NN_Inputs + NN_Outputs

BinaryIndexes = [0,5,6,15,16,17,18,19,20,23,26]
