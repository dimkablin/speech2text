#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <sys/types.h>
#include <unistd.h>

int main() {
	int result_count = 0;

	struct dirent **namelist;
	int n;

	n = scandir("/proc/", &namelist, NULL, alphasort);
	if (n == -1) {
		perror("scandir");
		exit(EXIT_FAILURE);
	}

	while (n--) {
		if (isdigit(namelist[n]->d_name[0])) {
			char buff[256] = "";
			strcat(buff, "/proc/");
			strcat(buff, namelist[n]->d_name);
			strcat(buff, "/status");

			FILE* f = fopen(buff, "r");
			char name[256];

			fscanf(f, "%*s %s", name);

			if (strcmp(name, "genenv") == 0) {
				++result_count;
			}

			fclose(f);
		}

		free(namelist[n]);
	}

	printf("%i\n", result_count);

	free(namelist);
	exit(EXIT_SUCCESS);
}