/*
    @Positive
 * Copyright (c) 2003, 2019, Oracle and/or its affiliates. All rights reserved.
    @Positive
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @Positive
 * This code is free software; you can redistribute it and/or modify it
    @Positive
 * under the terms of the GNU General Public License version 2 only, as
    @Positive
 * published by the Free Software Foundation.  Oracle designates this
    @Positive
 * particular file as subject to the "Classpath" exception as provided
    @Positive
 * by Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @Positive
 * This code is distributed in the hope that it will be useful, but WITHOUT
    @Positive
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @Positive
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @Positive
 * version 2 for more details (a copy is included in the LICENSE file that
    @Positive
 * accompanied this code).
    @Positive
 *
    @Positive
 * You should have received a copy of the GNU General Public License version
    @Positive
 * 2 along with this work; if not, write to the Free Software Foundation,
    @Positive
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Positive
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @Positive
 * or visit www.oracle.com if you need additional information or have any
    @Positive
 * questions.
    @Positive
 */
    @Positive
package java.lang.management;

    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.util.Map;

    @Positive
@AnnotatedFor({ "nullness" })
    @Positive
public interface ThreadMXBean extends PlatformManagedObject {

    @Positive
    public int getThreadCount();

    @Positive
    public int getPeakThreadCount();

    @Positive
    public long getTotalStartedThreadCount();

    @Positive
    public int getDaemonThreadCount();

    @Positive
    public long[] getAllThreadIds();

    @Positive
    @Nullable
    @Positive
    public ThreadInfo getThreadInfo(long id);

    @Positive
    @Nullable
    @Positive
    public ThreadInfo[] getThreadInfo(long[] ids);

    @Positive
    @Nullable
    @Positive
    public ThreadInfo getThreadInfo(long id, int maxDepth);

    @Positive
    @Nullable
    @Positive
    public ThreadInfo[] getThreadInfo(long[] ids, int maxDepth);

    @Positive
    public boolean isThreadContentionMonitoringSupported();

    @Positive
    public boolean isThreadContentionMonitoringEnabled();

    @Positive
    public void setThreadContentionMonitoringEnabled(boolean enable);

    @Positive
    public long getCurrentThreadCpuTime();

    @Positive
    public long getCurrentThreadUserTime();

    @Positive
    public long getThreadCpuTime(long id);

    @Positive
    public long getThreadUserTime(long id);

    @Positive
    public boolean isThreadCpuTimeSupported();

    @Positive
    public boolean isCurrentThreadCpuTimeSupported();

    @Positive
    public boolean isThreadCpuTimeEnabled();

    @Positive
    public void setThreadCpuTimeEnabled(boolean enable);

    @Positive
    public long @Nullable [] findMonitorDeadlockedThreads();

    @Positive
    public void resetPeakThreadCount();

    @Positive
    public long @Nullable [] findDeadlockedThreads();

    @Positive
    public boolean isObjectMonitorUsageSupported();

    @Positive
    public boolean isSynchronizerUsageSupported();

    @Positive
    @Nullable
    @Positive
    public ThreadInfo[] getThreadInfo(long[] ids, boolean lockedMonitors, boolean lockedSynchronizers);

    @Positive
    public default ThreadInfo[] getThreadInfo(long[] ids, boolean lockedMonitors, boolean lockedSynchronizers, int maxDepth);

    @Positive
    public ThreadInfo[] dumpAllThreads(boolean lockedMonitors, boolean lockedSynchronizers);

    @Positive
    public default ThreadInfo[] dumpAllThreads(boolean lockedMonitors, boolean lockedSynchronizers, int maxDepth);
    @Positive
}

// CFWR semantic augmentation - variant 0
