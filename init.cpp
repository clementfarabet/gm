#include "TH.h"
#include "luaT.h"

#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define torch_string_(NAME) TH_CONCAT_STRING_3(torch., Real, NAME)
#define gm_(NAME) TH_CONCAT_3(gm_, Real, NAME)
#define gm_energies_(NAME) TH_CONCAT_3(gm_energies_, Real, NAME)
#define gm_infer_(NAME) TH_CONCAT_3(gm_infer_, Real, NAME)

static const void* torch_FloatTensor_id = NULL;
static const void* torch_DoubleTensor_id = NULL;

#include "generic/gm.c"
#include "THGenerateFloatTypes.h"

#include "generic/gm_infer.c"
#include "THGenerateFloatTypes.h"

#include "generic/gm_energies.c"
#include "THGenerateFloatTypes.h"

extern "C" {
  DLL_EXPORT int luaopen_libgm(lua_State *L)
  {
    torch_FloatTensor_id = luaT_checktypename2id(L, "torch.FloatTensor");
    torch_DoubleTensor_id = luaT_checktypename2id(L, "torch.DoubleTensor");

    gm_FloatInit(L);
    gm_DoubleInit(L);

    gm_energies_FloatInit(L);
    gm_energies_DoubleInit(L);

    gm_infer_FloatInit(L);
    gm_infer_DoubleInit(L);

    return 1;
  }
}
