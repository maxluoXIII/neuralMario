function press(button)
	local toPress = {};
	toPress[button] = true;
	joypad.set(toPress, 1);
	emu.frameadvance();
	toPress[button] = null;
	joypad.set(toPress, 1);
	emu.frameadvance();
end

function delay(time)
	for i = 1, time do
		emu.frameadvance();
	end
end

function enterLevel()
	for i = 1, 3 do
		press("A");
		delay(60);
	end
	delay(120);
	press("Right");
	delay(60);
	press("A");
	emu.frameadvance();
end

enterLevel();
buttons = { };
buttons["Right"] = true;
buttons["A"] = true;

testForm = forms.newform(200, 260, "Test");

while true do
	if buttons["A"] == true then
		buttons["A"] = false;
	else
		buttons["A"] = true;
	end
	joypad.set(buttons, 1);
	emu.frameadvance()
end
