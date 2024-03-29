// Fill out your copyright notice in the Description page of Project Settings.


#include "point_represtation.h"
#include "GameFramework/Actor.h"
#include <iostream>
#include <fstream>
#include "HAL/PlatformFileManager.h"
using namespace std;
// "GameFramework/BasicClasses.h"

static int id_counter = 0;
static bool generated_point_array = false;
static double point_array[1000][24][3];
static int number_of_frames = 144;


// Sets default values
Apoint_represtation::Apoint_represtation()
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

	//id_counter++;
	//id = id_counter - 1;
	
	//Sphere1 = CreateDefaultSubobject<USphereComponent>(this, TEXT("Sphere1"));
	//Sphere1->InitSphereRadius(250.0f);

}

// Called when the game starts or when spawned
void Apoint_represtation::BeginPlay()
{
	Super::BeginPlay();

	id = id_counter;
	id_counter++;

	if (!generated_point_array) {

		generated_point_array = true;

		UE_LOG(LogTemp, Warning, TEXT("The first element of the array is %i"), point_array[0][0][0]);

		UE_LOG(LogTemp, Warning, TEXT("The id of this object is %i"), id);
	}
	
	//UE_LOG(LogTemp, Warning, TEXT("The id of this object is %i"), id);
	
}

// Called every frame
void Apoint_represtation::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

	//UE_LOG(LogTemp, Warning, TEXT("The DeltaTime currently is %i"), DeltaTime);

	frame++;
	/*if (frame >= 144) {
		frame = 0;
	}*/

	if (id % 22 == 0) {
		if (frame % 200 == 0) {
			UE_LOG(LogTemp, Warning, TEXT("The id of this object is %i"), id);

			//UE_LOG(LogTemp, Warning, TEXT("The size of the point array is %i"), sizeof(point_array)/sizeof(point_array[0]));


			string filepath = "../Animations/Anim.rec";

			FString FileData;
			//FFileHelper::EHashOptions Verification;
			FString FullFilePath = FPaths::ProjectContentDir() + FString(filepath.c_str());
			FFileHelper::LoadFileToString(FileData, *FullFilePath);

			//UE_LOG(LogTemp, Warning, TEXT("The id of this object is %s"), *FileData);

			string FileString = TCHAR_TO_UTF8(*FileData);

			string temp_string = "";

			int AFrame = 0;
			int Point = 0;
			int coordinate = 0;

			int j = 0;

			//char temp_chars[] = {"a"};

			for (int i = 0; i < FileString.size(); i++) {
				if (i < FileString.size() - 1) {
					//temp_chars = { FileString[i], FileString[i + 1] };
					if (FileString[i] == *":") {
						//UE_LOG(LogTemp, Warning, TEXT("Found a line break"));

						if (FileString[i + 2] == *";" || FileString[i + 3] == *";" || FileString[i + 4] == *";") {
							AFrame++;
							Point = 0;
						}
						else {
							if (FileString[i -1] == *"l" && FileString[i - 2] == *"a") {
								//UE_LOG(LogTemp, Warning, TEXT("found al"));
							}
							else if (FileString[i - 1] == *"p" && FileString[i - 2] == *"i" && FileString[i - 3] == *"T") {
								//UE_LOG(LogTemp, Warning, TEXT("found tip"));
							}
							else if (FileString[i - 1] == *"e" && FileString[i - 2] == *"s" && FileString[i - 3] == *"o") {
								//UE_LOG(LogTemp, Warning, TEXT("found ose"));
							}
							else if (FileString[i - 1] == *"e" && FileString[i - 2] == *"t" && FileString[i - 3] == *"a") {
								//UE_LOG(LogTemp, Warning, TEXT("found ate"));
							}
							else if (FileString[i - 1] == *"r" && FileString[i - 2] == *"e") {
								//UE_LOG(LogTemp, Warning, TEXT("found ate"));
							}
							else if (FileString[i - 1] == *"2" && FileString[i - 2] == *"0" && FileString[i - 3] == *"k") {
								//UE_LOG(LogTemp, Warning, TEXT("found ate"));
							}
							else if (FileString[i - 1] == *"e" && FileString[i - 2] == *"l" && FileString[i - 3] == *"c") {
								//UE_LOG(LogTemp, Warning, TEXT("found ate"));
							}
							else {
								temp_string = FileString.substr(i + 1, 8);
								//UE_LOG(LogTemp, Warning, TEXT("point number %s"), UTF8_TO_TCHAR(temp_string.c_str()));
								if (Point <= 22) {
									point_array[AFrame][Point - 1][0] = stod(temp_string);
								}


								bool found_comma = false;

								j = 0;

								while (!found_comma) {
									if (FileString[i + j] == *",") {
										found_comma = true;
									}
									j++;
								}

								temp_string = FileString.substr(i + j, 8);
								//UE_LOG(LogTemp, Warning, TEXT("point number %s"), UTF8_TO_TCHAR(temp_string.c_str()));
								if (Point <= 22) {
									point_array[AFrame][Point - 1][1] = stod(temp_string);
								}

								found_comma = false;

								while (!found_comma) {
									if (FileString[i + j] == *",") {
										found_comma = true;
									}
									j++;
								}

								temp_string = FileString.substr(i + j, 8);
								//UE_LOG(LogTemp, Warning, TEXT("point number %s"), UTF8_TO_TCHAR(temp_string.c_str()));
								if (Point <= 22) {
									point_array[AFrame][Point - 1][2] = stod(temp_string);
								}

								Point++;
							}
						}
					}
				}
			}
			number_of_frames = AFrame;
		}
	}

	if (id < 22) {
		//FVector coords;
		coords.X = point_array[frame % number_of_frames][id][0]; //* 100;
		coords.Y = point_array[frame % number_of_frames][id][1]; //* 100;
		coords.Z = point_array[frame % number_of_frames][id][2]; //* 100;
		// double coords[3] = point_array[0][id];
		// SetActorLocation(coords);
	}
	else {
		coords.X = point_array[frame % number_of_frames][id % 22][0]; //* 100;
		coords.Y = point_array[frame % number_of_frames][id % 22][1];//* 100;
		coords.Z = point_array[frame % number_of_frames][id % 22][2]; //* 100;
	}
	//UE_LOG(LogTemp, Warning, TEXT("The id of this object is %i"), id);
}

