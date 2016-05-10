-- neuralMario by Maxiwell Luo, Srisri Soundararajan, and Shahzar Rizvi
-- For the fourth quarter project of Computer Science Topics 2015-2016 taught by Dr. Lopykinski

--[[ Until the ending comment, the following code was copied from SethBling's MarI/O.
		This code was copied because retrieving inputs from the emulator is out of the scope of this project.--]]
Filename = "save.state";
ButtonNames = {"A", "B", "X", "Y", "Up", "Down", "Left", "Right"};

BoxRadius = 6;
InputSize = (BoxRadius*2+1)*(BoxRadius*2+1);

Inputs = InputSize+1;
Outputs = #ButtonNames;

Population = 300;
DeltaDisjoint = 2.0;
DeltaWeights = 0.4;
DeltaThreshold = 1.0;

StaleSpecies = 15;

MutateConnectionsChance = 0.25;
PerturbChance = 0.90;
CrossoverChance = 0.75;
LinkMutationChance = 2.0;
NodeMutationChance = 0.50;
BiasMutationChance = 0.40;
StepSize = 0.1;
DisableMutationChance = 0.4;
EnableMutationChance = 0.2;

TimeoutConstant = 20;

MaxNodes = 1000000;

-- Finds the positions of things on the screen at one moment??
function getPositions()
        if gameinfo.getromname() == "Super Mario World (USA)" then
                marioX = memory.read_s16_le(0x94);
                marioY = memory.read_s16_le(0x96);

                local layer1x = memory.read_s16_le(0x1A);
                local layer1y = memory.read_s16_le(0x1C);

                screenX = marioX-layer1x;
                screenY = marioY-layer1y;
        elseif gameinfo.getromname() == "Super Mario Bros." then
                marioX = memory.readbyte(0x6D) * 0x100 + memory.readbyte(0x86);
                marioY = memory.readbyte(0x03B8)+16;

                screenX = memory.readbyte(0x03AD);
                screenY = memory.readbyte(0x03B8);
        end
end

function getTile(dx, dy)
        if gameinfo.getromname() == "Super Mario World (USA)" then
                x = math.floor((marioX+dx+8)/16);
                y = math.floor((marioY+dy)/16);

                return memory.readbyte(0x1C800 + math.floor(x/0x10)*0x1B0 + y*0x10 + x%0x10);
        elseif gameinfo.getromname() == "Super Mario Bros." then
                local x = marioX + dx + 8;
                local y = marioY + dy - 16;
                local page = math.floor(x/256)%2;

                local subx = math.floor((x%256)/16);
                local suby = math.floor((y - 32)/16);
                local addr = 0x500 + page*13*16+suby*16+subx;

                if suby >= 13 or suby < 0 then
                        return 0;
                end

                if memory.readbyte(addr) ~= 0 then
                        return 1;
                else
                        return 0;
                end
        end
end

function getSprites()
        if gameinfo.getromname() == "Super Mario World (USA)" then
                local sprites = {};
                for slot=0,11 do
                        local status = memory.readbyte(0x14C8+slot);
                        if status ~= 0 then
                                spritex = memory.readbyte(0xE4+slot) + memory.readbyte(0x14E0+slot)*256;
                                spritey = memory.readbyte(0xD8+slot) + memory.readbyte(0x14D4+slot)*256;
                                sprites[#sprites+1] = {["x"]=spritex, ["y"]=spritey};
                        end
                end

                return sprites;
        elseif gameinfo.getromname() == "Super Mario Bros." then
                local sprites = {};
                for slot=0,4 do
                        local enemy = memory.readbyte(0xF+slot);
                        if enemy ~= 0 then
                                local ex = memory.readbyte(0x6E + slot)*0x100 + memory.readbyte(0x87+slot);
                                local ey = memory.readbyte(0xCF + slot)+24;
                                sprites[#sprites+1] = {["x"]=ex,["y"]=ey};
                        end
                end

                return sprites;
        end
end

function getExtendedSprites()
        if gameinfo.getromname() == "Super Mario World (USA)" then
                local extended = {};
                for slot=0,11 do
                        local number = memory.readbyte(0x170B+slot);
                        if number ~= 0 then
                                spritex = memory.readbyte(0x171F+slot) + memory.readbyte(0x1733+slot)*256;
                                spritey = memory.readbyte(0x1715+slot) + memory.readbyte(0x1729+slot)*256;
                                extended[#extended+1] = {["x"]=spritex, ["y"]=spritey};
                        end
                end

                return extended;
        elseif gameinfo.getromname() == "Super Mario Bros." then
                return {};
        end
end

function getInputs()
        getPositions();

        sprites = getSprites();
        extended = getExtendedSprites();

        local inputs = {};

        for dy=-BoxRadius*16,BoxRadius*16,16 do
                for dx=-BoxRadius*16,BoxRadius*16,16 do
                        inputs[#inputs+1] = 0;

                        tile = getTile(dx, dy);
                        if tile == 1 and marioY+dy < 0x1B0 then
                                inputs[#inputs] = 1;
                        end

                        for i = 1,#sprites do
                                distx = math.abs(sprites[i]["x"] - (marioX+dx));
                                disty = math.abs(sprites[i]["y"] - (marioY+dy));
                                if distx <= 8 and disty <= 8 then
                                        inputs[#inputs] = -1;
                                end
                        end

                        for i = 1,#extended do
                                distx = math.abs(extended[i]["x"] - (marioX+dx));
                                disty = math.abs(extended[i]["y"] - (marioY+dy));
                                if distx < 8 and disty < 8 then
                                        inputs[#inputs] = -1;
                                end
                        end
                end
        end

        --mariovx = memory.read_s8(0x7B)
        --mariovy = memory.read_s8(0x7D)

        return inputs;
end

--[[ The code above is copied from SethBling's MarI/O --]]

function sigmoid(x)
	return 1 / (1 + math.exp(-4.9 * x));
end

function newNeuron()
	local neuron = {};
	neuron.id = 0;
	neuron.neuronType = -1;
	-- input = 0, hidden = 1, output = 2, bias = 3
	neuron.isRecurrent = false;
	neuron.activationResponse = 0.0;
	neuron.splitX = 0.0;
	neuron.splitY = 0.0;
	neuron.incoming = {};
	return neuron;
end

function newLink()
	local link = {};
	link.enter = 0;
	link.out = 0;
	link.weight = 0.0;
	link.id = 0;
	link.enabled = true;
	link.isRecurrent = false;
	return link;
end

function newGenome()
	local genome = {};
	genome.neurons = {};
	genome.links = {};
	genome.network = {};
	genome.maxneuron = 0;
	genome.rawFitness = 0.0;
	genome.adjustedFitness = 0.0;
	genome.amountToSpawn = 0.0;
	genome.numInputs = 0;
	genome.numOutputs = 0;
	genome.species = 0;
	genome.mutationRates = {};
	genome.mutationRates["connections"] = MutateConnectionsChance;
	genome.mutationRates["link"] = LinkMutationChance;
	genome.mutationRates["bias"] = BiasMutationChance;
	genome.mutationRates["node"] = NodeMutationChance;
	genome.mutationRates["enable"] = EnableMutationChance;
	genome.mutationRates["disable"] = DisableMutationChance;
	genome.mutationRates["step"] = StepSize;
	return genome;
end

function createNetwork(genome)
	local network = {};
	network.neurons = {};
	-- The inputs from the screen
	for i = 1, Inputs do
		network.neuron[i] = newNeuron();
	end

	-- The buttons (Outputs)
	for i = 1, Outputs do
		network.neuron[MaxNodes + i] = newNeuron();
	end

	table.sort(genome.links, function(a, b) return a.out < b.out; end);

	-- Hidden nodes
	for i = 1, #genomes.genes do
		local link = genome.links[i];
		if link.enabled then
			if network.neurons[link.out] == nil then
				network.neurons[link.out] = newNeuron();
			end
			table.insert(network.neurons[link.out].incoming, link);
			if network.neurons[link.enter] == nil then
				network.neurons[link.enter] = newNeuron();
			end
		end
	end

	genome.network = network;

end

function evaluateNetwork(network, inputs)
	table.insert(inputs, 1);
	if #inputs ~= Inputs then
		console.writeline("Inputs don't match");
	end

	for local i = 1, #network.neurons do
		local sum = 0;
		for local j = 1, #network.neurons[i].incoming do
			sum = sum + network.neurons[i].incoming[j].weight * network.neurons[i].incoming[j].enter.value;
		end

		if #network.neurons[i].incoming > 0 then
			network.neurons[i].value = sigmoid(sum);
		end
	end

	local outputs = {};
	for i = 1, Outputs do
		if network.neurons[MaxNodes + i].value > .5 then
			outputs[ButtonNames[i]] = true;
		else
			outputs[ButtonNames[i]] = false;
		end
	end

	return outputs;
end

while true do

end
