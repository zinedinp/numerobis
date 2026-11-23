//! Synchronisation and queries.

use super::{Area, Op, Spec, core, core10, ext};
use crate::vulkan::ApiVersion as V;
use crate::vulkan::features::Chain;

pub const SPECS: &[Spec] = &[
    Spec {
        op: Op::TimelineSemaphore,
        area: Area::Sync,
        name: "timeline_semaphore",
        doc: "semaphores with a 64-bit counter, waitable from the host",
        routes: &[
            core(V::V1_2, Chain::Vulkan12, &["timelineSemaphore"]),
            ext(
                c"VK_KHR_timeline_semaphore",
                &[],
                Some(Chain::TimelineSemaphore),
                &["timelineSemaphore"],
            ),
        ],
    },
    Spec {
        op: Op::Synchronization2,
        area: Area::Sync,
        name: "synchronization2",
        doc: "vkCmdPipelineBarrier2 and vkQueueSubmit2",
        routes: &[
            core(V::V1_3, Chain::Vulkan13, &["synchronization2"]),
            ext(
                c"VK_KHR_synchronization2",
                &[],
                Some(Chain::Synchronization2),
                &["synchronization2"],
            ),
        ],
    },
    Spec {
        op: Op::HostQueryReset,
        area: Area::Sync,
        name: "host_query_reset",
        doc: "vkResetQueryPool from the CPU, no reset commands",
        routes: &[
            core(V::V1_2, Chain::Vulkan12, &["hostQueryReset"]),
            ext(
                c"VK_EXT_host_query_reset",
                &[],
                Some(Chain::HostQueryReset),
                &["hostQueryReset"],
            ),
        ],
    },
    Spec {
        op: Op::PipelineStatistics,
        area: Area::Sync,
        name: "pipeline_statistics",
        doc: "pipeline statistics queries",
        routes: &[core10(&["pipelineStatisticsQuery"])],
    },
];
