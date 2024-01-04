/*
    Shaheer Ziya (**********)
    Developed & Tested on Ubuntu 20.04.6 LTS

    This program is a shell that can execute commands with arguments and pipes.
    The shell meets all requiremnts as long as the assumptions made in the assignment remain valid.
    Inputting an invalid command with pipes e.g. "ls -la | simp" or "ls | exit | ls" was not expicitly
    asked to be handled, so the program in that case does provide statistics since the requirements
    remain ambiguous. Also for the sake of my eyes, I print an extra new line before before and after
    prompts and statistics in this shell.

*/

#include <stdio.h>              // For scanf() & printf()
#include <string.h>             // For strncmp() etc.
#include <ctype.h>              // For isspace()
#include <unistd.h>             // For fork(), exec() family, getpid(), pipe() etc.
#include <stdlib.h>             // For strtol, exit() etc.
#include <sys/wait.h>           // For wait() and waitpid()
#include <sys/types.h>          // For pid_t

#define MAX_LINE_LENGTH 256
#define NUMCMDS 5

#define PIPE_READ 0
#define PIPE_WRITE 1

// Global Flag Variables
volatile sig_atomic_t SIGUSR1received = 0;  // Flag to indicate if SIGUSR1 has been received
volatile sig_atomic_t childRunning = 0;     // Flag to indicate if a child process is running

// When the user presses Ctrl+C (SIGINT), print a new line and the shell header; Flush the output buffer
void SIGINT_Handler(int signum) {
    if (childRunning) {
        // Do nothing
    } else {
        printf("\n## JCshell [%d] ##  ", getpid());
        fflush(stdout);
    }
}
// When SIGUSR1 is received, set the flag to indicate that SIGUSR1 has been received
void SIGUSR1_Handler(int signum) {
    SIGUSR1received = 1;
}

void sliceString(const char *src, int startIndex, int endIndex, char *dest);
void removeWhitespace(char *str);
char* trimSpace(char *str);

void parseInput(char *text, char cmds[NUMCMDS][1024]);
void parseCommand(char *text, char *cmd, char **args);


/* A function that obtains the statistics of a given process from proc/{pid}/stat and proc/{pid}/status
and stores the first half (before the exit code or signal) in the first index of the returned array and the
second half (after the exit code or signal) in the second index of the returned array.

It is important to note that the returned array is dynamically allocated and must be freed after use.
It is also important to note that this function must be called after the child process has been turned
into a zombie process (i.e. after waitid() but before waitpid())

    @param childPID: The PID of the child process
    @return: An array (of size 2) of strings containing the statistics of the child process
*/
char** getRunStats(pid_t childPID) {
    // This function opens proc/$$/stat & proc/$$/status to retrieve the neccessary information and prints it.
    
    /*~~~Open and read the files, storing the information in local variables~~~*/
    FILE *file;
    char str[50];               // Hold the path to the proc file
    
    char state[50];             // Hold the state of the process
    char cmdPar[50], cmd[50];   // Hold the name of the command in brackets e.g. (ls) and the name of the command e.g. ls
    int parentPID;              // Hold the parent PID of the process
    unsigned long h, ut, st;    // Hold data for time calculations
    int X;                      // Placeholder for irrelevant data (4bytes)

    // Read the process info from the proc/$$/stat file
    sprintf(str, "/proc/%d/stat", (int)childPID);
    file = fopen(str, "r");
   
    if (file == NULL)  {
        printf("Error in opening proc/%d/stat file for [%d]\n", childPID, childPID);
        exit(1);
    }

    fscanf(file, "%d %s %c %d %d %d %d %d %u %lu %lu %lu %lu %lu %lu",
            &X, cmdPar, &state[0], &parentPID, &X, &X, &X, &X, (unsigned *)&X, &h, &h, &h, &h, &ut, &st);

    fclose(file);

    // Access proc/$$/status
    sprintf(str, "/proc/%d/status", (int)childPID);
    file = fopen(str, "r");

    if (file == NULL)  {
        printf("Error in opening proc/%d/status file for [%d]\n", childPID, childPID);
        exit(1);
    }

    // Read the context switch info from the proc/$$/status file
    char line1[MAX_LINE_LENGTH], line2[MAX_LINE_LENGTH], currentLine[MAX_LINE_LENGTH];

    // Read each line from the file until the end
    while (fgets(currentLine, MAX_LINE_LENGTH, file) != NULL) {
        // Move line2 to line1 and currentLine to line2
        strcpy(line1, line2);
        strcpy(line2, currentLine);
    }

    fclose(file);

    /*~~~Format the retrieved information as neccessary and return it~~~*/
    // Find the integer values of the context switches (We know there will be integers in the last two lines)
    long vctx, nvctx;

    sscanf(line1, "%*s %ld", &vctx);
    sscanf(line2, "%*s %ld", &nvctx);

    // Remove the brackets from the command name
    sliceString(cmdPar, 1, strlen(cmdPar) - 2, cmd);
    // Prepare the statistics to be printed
    char statP1[200];
    sprintf(statP1, "(PID)%d (CMD)%s (STATE)%c", (int)childPID, cmd, state[0]);
    char statP2[200];
    sprintf(statP2, "(PPID)%d (USER)%.2lf (SYS)%.2lf (VCTX)%ld (NVCTX)%ld",
        parentPID, // PPID
        ut * 1.0f / sysconf(_SC_CLK_TCK), st * 1.0f / sysconf(_SC_CLK_TCK), // USER, SYS
        vctx, nvctx // VCTX, NVCTX
    );

    // Allocate memory for the array of strings to be returned
    char** stats = malloc(2 * sizeof(char*));
    stats[0] = statP1;
    stats[1] = statP2;

    return stats;
}


/* A function that executes a given command with the given arguments and redirects the input and output to relevant pipes
if necessary. In case there is only a single command, no pipe magic occurs. This function is only called by the child process

    @param cmd: The command to be executed
    @param args: The arguments to be passed to the command
    @param fd: The file descriptors of the pipes
    @param cmdIdx: The index of the command in the input
    @param numPipes: The number of pipes in the input
*/
int runCommand(char *cmd, char *const *args, int fd[][2], int cmdIdx, int numPipes) {
    // Wait until SIGUSR1 is received
    while (!SIGUSR1received) {
        pause();
    }
    // Reset the flag
    SIGUSR1received = 0;

    // If not the last command write output to the current pipe
    if (cmdIdx < numPipes) { 
        dup2(fd[cmdIdx][PIPE_WRITE], STDOUT_FILENO);
    }

    // If not the first command read the input from the previous pipe
    if (cmdIdx > 0) { 
        dup2(fd[cmdIdx - 1][PIPE_READ], STDIN_FILENO);
    }

    // Close all the pipe file descriptors for this child process
    for (int i = 0; i < numPipes; i++) {
        close(fd[i][PIPE_READ]);
        close(fd[i][PIPE_WRITE]);
    }

    execvp(cmd, args);
    // If execvp() fails, print error message and exit
    printf("\nJCshell: '%s': No such file or directory", cmd);
    kill(getppid(), SIGUSR1); // Send SIGUSR1 to the parent process to indicate that execvp() failed
    exit(2); // Exit with code 2 to indicate a commandline error
}


int runPipes(char cmds[NUMCMDS][1024]) {
    
    // Let numCmds be the number of commands in the input, then there must exist (numCmds - 1) pipes
    int numPipes = -1;
    while(cmds[numPipes + 1][0] != '\0') {
        numPipes++;
    } 
    
    int numCmds = numPipes + 1;

    // Create the all the pipes
    int fd[numPipes][2];  // Store the file descriptors of the pipes 
    /* Assign two slots in the array for each pipe (one for read and one for write)
    For example, if numPipes = 2, fd looks like "[r1, w1, r2, w2]" */
    for (int i = 0; i < numPipes; i++) {
        if (pipe(fd[i]) < 0) {
            perror("pipe");
            return -1;
        }
    }

    int PIDs[numPipes]; // The array to store the PIDs of the child processes
    // Run all the commands in the input by spawning child processes that communicate through pipes
    for (int cmdIdx = 0; cmdIdx < numCmds; cmdIdx++) {
        
        PIDs[cmdIdx] = fork();

        if (PIDs[cmdIdx] < 0) {
            perror("Error: Fork Failed\n");
            exit(1);
        } else if (PIDs[cmdIdx] == 0) { // Child Process
            // Parse the command and execute it
            char cmd[1024];
            char *args[30] = {};
            parseCommand(cmds[cmdIdx], cmd, args); // Parse the i-th cmd in the input and save it in the cmd and args variables
            runCommand(cmd, args, fd, cmdIdx, numPipes);
        }
    } 
    
    // Parent Process

    // Send SIGUSR1 signals to the child process (to start execution)
    for (int i = 0; i < numCmds; i++) {
        kill(PIDs[i], SIGUSR1);
    }

    // Close all the pipes for the parent process
    for (int i = 0; i < numPipes; i++) {
        close(fd[i][PIPE_READ]);
        close(fd[i][PIPE_WRITE]);
    }

    char stats[numCmds][200]; // Store the statistics of each child process
    siginfo_t info; // Needed to get pid of child process
    int status[numCmds]; // Store the exit status of each child process
    childRunning = 1; // Set the flag to indicate that children processes are running
    // Get the statistics of each child process in termination order
    for (int i = 0; i < numCmds; i++) {
        // Wait for a child process to turn into a zombie process (to access prcoess info)
        int ret = waitid(P_ALL, 0, &info, WNOWAIT | WEXITED); // Waits for any child process to change state
        if (!ret) {
            char** procStats = getRunStats(info.si_pid); // Store the statistics of the child process (info.si_pid <-> childPID)
            // Store the statistics in the stats array (in termination order)
            waitpid(info.si_pid, &status[i], 0); // Child process's resources have been cleaned up
            // Print the statistics based on the exit condition of the child process
            if (WIFEXITED(status[i])) { // Exited Voluntarily
                if (SIGUSR1received) {
                    *stats[i] = '\0'; // Set the string to be empty
                } else {
                    sprintf(stats[i], "%s (EXCODE)%d %s",
                        procStats[0], WEXITSTATUS(status[i]), procStats[1]);
                }
            } else if (WIFSIGNALED(status[i])) { // Exited due to a signal;
                sprintf(stats[i], "%s (EXSIG)%s %s",
                        procStats[0], strsignal(WTERMSIG(status[i])), procStats[1]);
            }
            free(procStats);
        } else {
            perror("Error: waitid() failed\n");
        }
    }
    
    childRunning = 0; // Set the flag to indicate that children processes are not running
    
    // Print the statistics in termination order
    if (!SIGUSR1received) {
        printf("\n");
    }
    for (int i = 0; i < numCmds; i++) {
        printf("%s\n", stats[i]);
    }
    printf("\n");

    SIGUSR1received = 0; // Reset the flag
}


int main() {
    signal(SIGINT, SIGINT_Handler); // Register the signal handler for SIGINT
    signal(SIGUSR1, SIGUSR1_Handler); // Register the signal handler for SIGUSR1

    // The commands (or input) never exceeds 1024 characters (with a max of 30 strings)
    char input[1025] = "";

    // The main loop only exits when exit without arguments is received
    while (1) {
        // Print the Shell Header and receive input
        printf("## JCshell [%d] ##  ", getpid());
        scanf(" %[^\n]", input); // Whitespace before % is important to ignore leading whitespace

        // Create a copy of the input to check for pipe errors
        char* inputCheck = malloc(1025 * sizeof(char));
        strcpy(inputCheck, input);
        removeWhitespace(inputCheck);
        // Check for pipe errors
        if (inputCheck[0] == '|') { // Check if the input starts with a pipe
            printf("\nJCshell: should not have pipe at start!\n\n");
            continue;
        } else if (inputCheck[strlen(inputCheck) - 1] == '|') { // Check if the input ends with a pipe
            printf("\nJCshell: should not have pipe at end!\n\n");
            continue;
        } else if (strstr(inputCheck, "||")) {
            printf("\nJCshell: should not have two | symbols without in-between command\n\n");
            continue;
        }

        free(inputCheck);

        // Parse the input into an array of commands from the input (separated by pipes)
        // There will be atmost 5 commands (in input) separated by pipes
        char comms[NUMCMDS][1024];
        parseInput(input, comms);

        // Parse the first command to check for exit
        char cmdCheck[1024];
        strcpy(cmdCheck, comms[0]); // parseCommand() modifies the input, so we use a copy
        char cmd[1024];
        char *args[30] = {};
        parseCommand(cmdCheck, cmd, args);      
        
        // Check if the exit commmand was input
        if (strcmp(cmd, "exit") == 0) {
            // Handle case where exit is input with arguments
            if (args[1] != NULL) {
                printf("\nJCshell: \"exit\" with too many arguments!!!\n\n");
                continue;
            } else {
                printf("\nJCshell: Terminated\n\n");
                exit(0);
            }
        }

        // Run the commands in the input
        runPipes(comms);
        
    }

}






/*~~~ String Functions ~~~*/

// A function that slices a string from a given start index to a given end index
void sliceString(const char *src, int startIndex, int endIndex, char *dest) {
    int length = endIndex - startIndex + 1;
    strncpy(dest, src + startIndex, length);
    dest[length] = '\0';
}

// A function that removes all whitespace from a string
void removeWhitespace(char *str) {
    char *src = str;
    char *dest = str;

    while (*src) {
        if (!isspace((unsigned char)*src)) {
            *dest++ = *src;
        }
        src++;
    }

    *dest = '\0'; // Null-terminate the modified string
}

// A function that trims the leading and trailing whitespace from a string
char* trimSpace(char *str) {
    char *end;
    /* Remove leading whitespace */
    while (isspace(*str)) {
        str = str + 1;
    }
    /* Remove trailing whitespace */
    end = str + strlen(str) - 1;
    while (end > str && isspace(*end)) {
        end = end - 1;
    }
    /* Terminate with null character */
    *(end + 1) = '\0';
    return str;
}

/*~~~ Text Splitting Commands ~~~*/

// A function that parses the input string into an array of strings separated by pipes
void parseInput(char *text, char cmds[NUMCMDS][1024]) { // It is assumed that there will be atmost 5 commands
    char *token;
    char *delim = "|";
    int index = 0;

    token = strtok(text, delim);
    while (token != NULL) {
        token = trimSpace(token);
        strcpy(cmds[index++], token);
        token = strtok(NULL, delim);
    }

    while (index < NUMCMDS) {
        cmds[index++][0] = '\0';
    }
}

/* A function that given a string of text containing the command, converts it into a format
that can be executed by the system
    @param text: The string of text to be parsed
    @param cmd: The command to be executed
    @param args: The arguments to be passed to the command
*/
void parseCommand(char *text, char *cmd, char **args) {

    char* token;
    char* delim = " ";
    int index = 0;

    token = strtok(text, delim);
    strcpy(cmd, token);

    while (token != NULL) {
        args[index] = token;
        
        token = strtok(NULL, delim);
        index++;
    }

}