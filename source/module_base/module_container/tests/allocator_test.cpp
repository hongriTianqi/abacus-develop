#include <vector>
#include <gtest/gtest.h>

#include "../tensor.h"
#include "../cpu_allocator.h"

TEST(CPUAllocator, AllocateAndFree) {
  container::CPUAllocator alloc;
  // Allocate memory of size 100.
  void* ptr = alloc.allocate(100);
  EXPECT_NE(nullptr, ptr);
  alloc.free(ptr);

  // Allocate memory of size 200 with alignment 16.
  ptr = alloc.allocate(200, 16);
  EXPECT_NE(nullptr, ptr);
  alloc.free(ptr);

  // Allocate memory of size 200 with alignment 16.
  ptr = alloc.allocate(0, 0);
  EXPECT_EQ(nullptr, ptr);
}

TEST(CPUAllocator, AllocatedSize) {
  container::CPUAllocator alloc;
  // Allocate memory of size 100 and check its size.
  void* ptr = alloc.allocate(100);
  EXPECT_NE(nullptr, ptr);
  alloc.free(ptr);
  ASSERT_EXIT(
      alloc.AllocatedSize(ptr),
      ::testing::ExitedWithCode(EXIT_FAILURE),
      "Method `AllocatedSize` not implemented!"
  );
}

TEST(CPUAllocator, GetDeviceType) {
  container::CPUAllocator alloc;
  EXPECT_EQ(container::DeviceType::CpuDevice,
            alloc.GetDeviceType());
}