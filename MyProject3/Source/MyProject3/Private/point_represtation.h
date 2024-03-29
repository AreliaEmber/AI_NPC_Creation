// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "point_represtation.generated.h"

UCLASS()
class Apoint_represtation : public AActor
{
	GENERATED_BODY()
	
	//static int id_counter;

	public:	
		// Sets default values for this actor's properties
		Apoint_represtation();

		UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "myproject")
		int id = 0;

		UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "myproject")
		FVector coords;

		int frame = 0;

	protected:
		// Called when the game starts or when spawned
		virtual void BeginPlay() override;

	public:	
		// Called every frame
		virtual void Tick(float DeltaTime) override;

};
