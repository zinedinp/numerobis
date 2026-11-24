//! Synchronisation and queries.

use super::{Area, Spec, Support, core, core10, ext};
use crate::vulkan::ApiVersion as V;
use crate::vulkan::features::Chain;

const AREA: Area = Area::Sync;

pub const TIMELINE_SEMAPHORE: Spec = Spec {
    area: AREA,
    support: Support::WINDOWS_TESTED,
    needs: &[],
    name: "timeline_semaphore",
    doc: "semaphores with a 64-bit counter, waitable from the host",
    routes: &[
        core(V::V1_2, Chain::Vulkan12, &["timelineSemaphore"]),
        ext(
            c"VK_KHR_timeline_semaphore",
            Some(Chain::TimelineSemaphore),
            &["timelineSemaphore"],
        ),
    ],
};

pub const SYNCHRONIZATION2: Spec = Spec {
    area: AREA,
    support: Support::WINDOWS_TESTED,
    needs: &[],
    name: "synchronization2",
    doc: "vkCmdPipelineBarrier2 and vkQueueSubmit2",
    routes: &[
        core(V::V1_3, Chain::Vulkan13, &["synchronization2"]),
        ext(
            c"VK_KHR_synchronization2",
            Some(Chain::Synchronization2),
            &["synchronization2"],
        ),
    ],
};

pub const HOST_QUERY_RESET: Spec = Spec {
    area: AREA,
    support: Support::WINDOWS_TESTED,
    needs: &[],
    name: "host_query_reset",
    doc: "vkResetQueryPool from the CPU, no reset commands",
    routes: &[
        core(V::V1_2, Chain::Vulkan12, &["hostQueryReset"]),
        ext(
            c"VK_EXT_host_query_reset",
            Some(Chain::HostQueryReset),
            &["hostQueryReset"],
        ),
    ],
};

pub const PIPELINE_STATISTICS: Spec = Spec {
    area: AREA,
    support: Support::WINDOWS_TESTED,
    needs: &[],
    name: "pipeline_statistics",
    doc: "pipeline statistics queries",
    routes: &[core10(&["pipelineStatisticsQuery"])],
};

pub const CALIBRATED_TIMESTAMPS: Spec = Spec {
    area: AREA,
    support: Support::WINDOWS_TESTED,
    needs: &[],
    name: "calibrated_timestamps",
    doc: "GPU timestamps next to a host clock reading (frame pacing, profiling)",
    routes: &[
        ext(c"VK_KHR_calibrated_timestamps", None, &[]),
        ext(c"VK_EXT_calibrated_timestamps", None, &[]),
    ],
};
