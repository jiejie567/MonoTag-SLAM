/*	
 * File: Random.cpp
 * Project: DUtils library
 * Author: Dorian Galvez-Lopez
 * Date: April 2010
 * Description: manages pseudo-random numbers
 * License: see the LICENSE.txt file
 *
 */

#include "Random.h"
#include "Timestamp.h"
#include <cstdlib>
#include <cstdint>
#include <mutex>
#include <stdexcept>
using namespace std;

namespace {

// Park-Miller with the reference Darwin rand() zero-seed convention.
// Deliberately keep its range mapping rather than changing RANSAC samples.
struct PortableRandomState
{
  uint32_t value = 0;
  bool already_seeded = false;
  mutex lock;
};

PortableRandomState& portableState()
{
  static PortableRandomState state;
  return state;
}

}

bool DUtils::Random::m_already_seeded = false;

bool DUtils::Random::PortableRandomEnabled()
{
  static const bool enabled = []() {
    const char* option = getenv("ORB_SLAM3_PORTABLE_RANDOM");
    return option && option[0]=='1' && option[1]=='\0';
  }();
  return enabled;
}

unsigned int DUtils::Random::PortableRandomDraw()
{
  PortableRandomState& state = portableState();
  lock_guard<mutex> guard(state.lock);
  int64_t value = state.value;
  if(value == 0) value = 123459876;
  const int64_t high = value / 127773;
  const int64_t low = value % 127773;
  value = 16807 * low - 2836 * high;
  if(value < 0) value += 2147483647;
  state.value = static_cast<uint32_t>(value);
  return state.value;
}

void DUtils::Random::SeedRand(){
	Timestamp time;
	time.setToCurrentTime();
	if(PortableRandomEnabled())
	{
		PortableRandomState& state = portableState();
		lock_guard<mutex> guard(state.lock);
		state.value = static_cast<uint32_t>(time.getFloatTime());
		return;
	}
	srand((unsigned)time.getFloatTime()); 
}

void DUtils::Random::SeedRandOnce()
{
  if(PortableRandomEnabled())
  {
    PortableRandomState& state = portableState();
    lock_guard<mutex> guard(state.lock);
    if(!state.already_seeded)
    {
      Timestamp time;
      time.setToCurrentTime();
      state.value = static_cast<uint32_t>(time.getFloatTime());
      state.already_seeded = true;
    }
    return;
  }
  if(!m_already_seeded)
  {
    DUtils::Random::SeedRand();
    m_already_seeded = true;
  }
}

void DUtils::Random::SeedRand(int seed)
{
	if(PortableRandomEnabled())
	{
		PortableRandomState& state = portableState();
		lock_guard<mutex> guard(state.lock);
		state.value = static_cast<uint32_t>(seed);
		return;
	}
	srand(seed); 
}

void DUtils::Random::SeedRandOnce(int seed)
{
  if(PortableRandomEnabled())
  {
    PortableRandomState& state = portableState();
    lock_guard<mutex> guard(state.lock);
    if(!state.already_seeded)
    {
      state.value = static_cast<uint32_t>(seed);
      state.already_seeded = true;
    }
    return;
  }
  if(!m_already_seeded)
  {
    DUtils::Random::SeedRand(seed);
    m_already_seeded = true;
  }
}

int DUtils::Random::RandomInt(int min, int max){
	if(PortableRandomEnabled())
	{
		if(min > max) throw invalid_argument("RandomInt requires min <= max");
		const int64_t span = static_cast<int64_t>(max) - min + 1;
		const int64_t offset = static_cast<int64_t>(
			(static_cast<double>(PortableRandomDraw()) / 2147483648.0) * span);
		return static_cast<int>(static_cast<int64_t>(min) + offset);
	}
	int d = max - min + 1;
	return int(((double)rand()/((double)RAND_MAX + 1.0)) * d) + min;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

DUtils::Random::UnrepeatedRandomizer::UnrepeatedRandomizer(int min, int max)
{
  if(min <= max)
  {
    m_min = min;
    m_max = max;
  }
  else
  {
    m_min = max;
    m_max = min;
  }

  createValues();
}

// ---------------------------------------------------------------------------

DUtils::Random::UnrepeatedRandomizer::UnrepeatedRandomizer
  (const DUtils::Random::UnrepeatedRandomizer& rnd)
{
  *this = rnd;
}

// ---------------------------------------------------------------------------

int DUtils::Random::UnrepeatedRandomizer::get()
{
  if(empty()) createValues();
  
  DUtils::Random::SeedRandOnce();
  
  int k = DUtils::Random::RandomInt(0, m_values.size()-1);
  int ret = m_values[k];
  m_values[k] = m_values.back();
  m_values.pop_back();
  
  return ret;
}

// ---------------------------------------------------------------------------

void DUtils::Random::UnrepeatedRandomizer::createValues()
{
  int n = m_max - m_min + 1;
  
  m_values.resize(n);
  for(int i = 0; i < n; ++i) m_values[i] = m_min + i;
}

// ---------------------------------------------------------------------------

void DUtils::Random::UnrepeatedRandomizer::reset()
{
  if((int)m_values.size() != m_max - m_min + 1) createValues();
}

// ---------------------------------------------------------------------------

DUtils::Random::UnrepeatedRandomizer& 
DUtils::Random::UnrepeatedRandomizer::operator=
  (const DUtils::Random::UnrepeatedRandomizer& rnd)
{
  if(this != &rnd)
  {
    this->m_min = rnd.m_min;
    this->m_max = rnd.m_max;
    this->m_values = rnd.m_values;
  }
  return *this;
}

// ---------------------------------------------------------------------------


