@echo off
echo START OF SCRIPT

for /L %%n in (1,1,24) do (
	echo Lead == %%n
	python -u run_gp.py --lead %%n --model gpy --dtrain 125 --dval 7 --dstart 0 --dend 243
)

echo END OF SCRIPT
