/*  Copyright 2011 AIT Austrian Institute of Technology
*
*   This file is part of OpenTLD.
*
*   OpenTLD is free software: you can redistribute it and/or modify
*   it under the terms of the GNU General Public License as published by
*    the Free Software Foundation, either version 3 of the License, or
*   (at your option) any later version.
*
*   OpenTLD is distributed in the hope that it will be useful,
*   but WITHOUT ANY WARRANTY; without even the implied warranty of
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*   GNU General Public License for more details.
*
*   You should have received a copy of the GNU General Public License
*   along with OpenTLD.  If not, see <http://www.gnu.org/licenses/>.
*
*/

/**
* @author Georg Nebehay
*/

#include "Main.h"
#include "Config.h"
#include "ImAcq.h"
#include "Gui.h"






using tld::Config;
using tld::Gui;
using tld::Settings;

#define __NO_STD_VECTOR
#define __CL_ENABLE_EXCEPTIONS

#include<CL/cl.h>
#include <string>
#include<iostream>
using namespace std;



#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <fstream>

#define SUCCESS 0
#define FAILURE 1

int convertToString(const char *filename, std::string& s)
{
	size_t size;
	char*  str;
	std::fstream f(filename, (std::fstream::in | std::fstream::binary));

	if (f.is_open())
	{
		size_t fileSize;
		f.seekg(0, std::fstream::end);
		size = fileSize = (size_t)f.tellg();
		f.seekg(0, std::fstream::beg);
		str = new char[size + 1];
		if (!str)
		{
			f.close();
			return 0;
		}

		f.read(str, fileSize);
		f.close();
		str[size] = '\0';
		s = str;
		delete[] str;
		return 0;
	}
	cout << "Error: failed to open file\n:" << filename << endl;
	return FAILURE;
}
int main(int argc, char **argv)
{



	Main *main = new Main();
	Config config;
	ImAcq *imAcq = imAcqAlloc();
	Gui *gui = new Gui();

	main->gui = gui;
	main->imAcq = imAcq;

	if (config.init(argc, argv) == PROGRAM_EXIT)
	{
		return EXIT_FAILURE;
	}

	config.configure(main);

	srand(main->seed);

	imAcqInit(imAcq);

	if (main->showOutput)
	{
		gui->init();
	}    main->doWork();

	delete main;
	main = NULL;
	delete gui;
	gui = NULL;


	return EXIT_SUCCESS;
}


