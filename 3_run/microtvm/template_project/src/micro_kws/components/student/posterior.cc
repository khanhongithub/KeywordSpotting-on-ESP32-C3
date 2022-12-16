#include <stdint.h>
#include <stdio.h>

#include "posterior.h"

#include "esp_log.h"
#include "esp_timer.h"

/**
 * @brief Default constructor for posterior handler
 *
 * @param history_length Number of past model outputs du consider.
 * @param trigger_threshold_single Threshold value between 0 and 255 for moving average.
 * @param suppression_ms For how many my a new detection should be ignored.
 * @param category_count Number of used labels.

 */
PosteriorHandler::PosteriorHandler(uint32_t history_length, uint8_t trigger_threshold_single,
                                   uint32_t suppression_ms, uint32_t category_count)
    : posterior_history_length_(history_length),
      posterior_trigger_threshold_(trigger_threshold_single * history_length),
      posterior_suppression_ms_(suppression_ms),
      posterior_category_count_(category_count) {

  /* ------------------------ */
  /* ENTER STUDENT CODE BELOW */
  /* ------------------------ */

  /*
   * Hints:
   * - data structured defined in (posterior.h) have to be initialized here
   * - Normally an embedded developer wouln;t use malloc() to dynamically allocate arrays etc.
   * - However to enable unit testing the history_length as well as the category_count are
   *   not constant and therefore allocation has to be done i.e. using malloc.
   * - While you are allowed to use C++ data structures, it is completely fine if you just
   *   use plain C arrays/pointers/...
   */

  /* ------------------------ */
  /* ENTER STUDENT CODE ABOVE */
  /* ------------------------ */
}

/**
 * @brief Destructor for posterior handler class
 */
PosteriorHandler::~PosteriorHandler() {

  /* ------------------------ */
  /* ENTER STUDENT CODE BELOW */
  /* ------------------------ */

  /*
   * Hints:
   * - Every data structure allocated in the constructor above has to be cleaned up properly
   * - This can for example be achieved using free()
   */

  /* ------------------------ */
  /* ENTER STUDENT CODE ABOVE */
  /* ------------------------ */
}

/**
 * @brief Implementation of posterior hanlding algorithm.
 *
 * @param new_posteriors The raw model outputs with unsigned 8-bit values.
 * @param time_ms Timestamp for posterior handling (ms).
 * @param top_category_index The index of the detected category/label returned by pointer.
 * @param trigger Flag which should be raised to true if a new detection is available.
 *
 * @return ESP_OK if no error occured.
 */
esp_err_t PosteriorHandler::Handle(uint8_t* new_posteriors, uint32_t time_ms,
                                   size_t* top_category_index, bool* trigger) {

  /* ------------------------ */
  /* ENTER STUDENT CODE BELOW */
  /* ------------------------ */

  /*
   * Hints:
   * - The goal is to implement a posterior handling algorithm descibed in Figure 2.1
   *   and section 2.2.1 of the Lab 2 manual.
   * - By using a moving average over the model outputs, we want to reduce the number
   *   of incorrect classifications i.e. caused by random spikes.
   * - If the calculated moving average for a class exceeds the trigger threshold a
   *   detection should be triggered (unless the deactivation period for a past detection
   *   is still active)
   * - The trigger should be raised for all classes (including silence and unknown) since
   *   the KeywordCallback method in backend.cc is reponsible for deciding which labels should
   *   be ignored.
   * - The supression time (in ms) defines the duration in which registered labels shall
   *   not trigger a new detection. (However their moving average should continue to be updated)
   * - Only if a detection was classified (outside of the deactivation period) the trigger argument
   *   should be set to true by the algorithm
   * - If trigger is high, the detected category index has to updated as well using the argument.
   * - You are allowed (and required) to introduce class variables inside include/posterior.h which
   * may than be (de-)initialized in the constructor/destuctor above.
   */

  /*
   * The following code is a basic example (not an accepted solution)
   * It just returns the class with the highest probablity.
   */

  size_t top_canidate_index = 0;
  uint8_t top_canidate_value = 0;
  for (size_t i = 0; i < posterior_category_count_; i++) {
    if (new_posteriors[i] > top_canidate_value) {
      top_canidate_index = i;
      top_canidate_value = new_posteriors[i];
    }
  }
  *trigger = true;  // no suppression
  *top_category_index = top_canidate_index;
  return ESP_OK;


  /* ------------------------ */
  /* ENTER STUDENT CODE ABOVE */
  /* ------------------------ */

  return ESP_OK;
}
