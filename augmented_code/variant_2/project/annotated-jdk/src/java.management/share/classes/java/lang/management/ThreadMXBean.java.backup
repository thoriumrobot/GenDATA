/*
 * Copyright (c) 2003, 2019, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */
package java.lang.management;

import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.util.Map;

@AnnotatedFor({ "nullness" })
public interface ThreadMXBean extends PlatformManagedObject {

    public int getThreadCount();

    public int getPeakThreadCount();

    public long getTotalStartedThreadCount();

    public int getDaemonThreadCount();

    public long[] getAllThreadIds();

    @Nullable
    public ThreadInfo getThreadInfo(long id);

    @Nullable
    public ThreadInfo[] getThreadInfo(long[] ids);

    @Nullable
    public ThreadInfo getThreadInfo(long id, int maxDepth);

    @Nullable
    public ThreadInfo[] getThreadInfo(long[] ids, int maxDepth);

    public boolean isThreadContentionMonitoringSupported();

    public boolean isThreadContentionMonitoringEnabled();

    public void setThreadContentionMonitoringEnabled(boolean enable);

    public long getCurrentThreadCpuTime();

    public long getCurrentThreadUserTime();

    public long getThreadCpuTime(long id);

    public long getThreadUserTime(long id);

    public boolean isThreadCpuTimeSupported();

    public boolean isCurrentThreadCpuTimeSupported();

    public boolean isThreadCpuTimeEnabled();

    public void setThreadCpuTimeEnabled(boolean enable);

    public long @Nullable [] findMonitorDeadlockedThreads();

    public void resetPeakThreadCount();

    public long @Nullable [] findDeadlockedThreads();

    public boolean isObjectMonitorUsageSupported();

    public boolean isSynchronizerUsageSupported();

    @Nullable
    public ThreadInfo[] getThreadInfo(long[] ids, boolean lockedMonitors, boolean lockedSynchronizers);

    public default ThreadInfo[] getThreadInfo(long[] ids, boolean lockedMonitors, boolean lockedSynchronizers, int maxDepth);

    public ThreadInfo[] dumpAllThreads(boolean lockedMonitors, boolean lockedSynchronizers);

    public default ThreadInfo[] dumpAllThreads(boolean lockedMonitors, boolean lockedSynchronizers, int maxDepth);
}
