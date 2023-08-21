#include "ShadowRemover.h"
#include <cstdio>
#include <ctime>
#include <dirent.h>
#include <string>
#include <iostream>
#include <algorithm>

using namespace std;

int main(int argc, char** argv) {

	// Check input
	// main ../dataset/Adobe/train/input/ ./
	if (argc != 3) {
		cout << "Usage: DocumentShadowRemoval.exe InputLocation OutputLocation" << endl; 
		getchar();
		return -1;
	}

	struct dirent *entry;
    DIR *dp;

    dp = opendir(argv[1]);
    if (dp == NULL) {
        perror("opendir: Path does not exist or could not be read.");
        return -1;
    }

    while ((entry = readdir(dp))) {
		string f_name = entry->d_name;
		transform(f_name.begin(), f_name.end(), f_name.begin(), ::toupper);

		if (f_name.find("JPG") != string::npos || f_name.find("PNG") != string::npos) {
			string src = argv[1];
			src += entry->d_name;

			string dst = argv[2];
			dst += entry->d_name;

			const int length1 = src.length();
  
			// declaring character array (+1 for null terminator)
			char* src_arr = new char[length1 + 1];
		
			// copying the contents of the
			// string to char array
			strcpy(src_arr, src.c_str());
			ShadowRemover* sr = new ShadowRemover(src_arr);
			
			const int length2 = dst.length();
  
			// declaring character array (+1 for null terminator)
			char* dst_arr = new char[length2 + 1];
		
			// copying the contents of the
			// string to char array
			strcpy(dst_arr, dst.c_str());
			
			sr->RemoveShadow(dst_arr);

			delete sr;
		}
	}
        

    closedir(dp);
    return 0;

	// Remove shadow
	// ShadowRemover* sr = new ShadowRemover(argv[1]);
	// sr->RemoveShadow(argv[2]);

	// delete sr;

	// return 0;

}

