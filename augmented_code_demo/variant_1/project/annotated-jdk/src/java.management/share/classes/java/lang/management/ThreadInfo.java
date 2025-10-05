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
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import javax.management.openmbean.ArrayType;
    @Positive
import javax.management.openmbean.CompositeData;
    @Positive
import sun.management.ManagementFactoryHelper;
    @Positive
import sun.management.ThreadInfoCompositeData;
    @Positive
import static java.lang.Thread.State.*;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public class ThreadInfo {

    @Positive
    public long getThreadId();

    @Positive
    public String getThreadName();

    @Positive
    public Thread.State getThreadState();

    @Positive
    public long getBlockedTime();

    @Positive
    public long getBlockedCount();

    @Positive
    public long getWaitedTime();

    @Positive
    public long getWaitedCount();

    @Positive
    public LockInfo getLockInfo();

    @Positive
    public String getLockName();

    @Positive
    public long getLockOwnerId();

    @Positive
    public String getLockOwnerName();

    @Positive
    public StackTraceElement[] getStackTrace();

    @Positive
    public boolean isSuspended();

    @Positive
    public boolean isInNative();

    @Positive
    public boolean isDaemon();

    @Positive
    public int getPriority();

    @Positive
    public String toString();

    @Positive
    public static ThreadInfo from(CompositeData cd);

    @Positive
    public MonitorInfo[] getLockedMonitors();

    @Positive
    public LockInfo[] getLockedSynchronizers();
    @Positive
}

// CFWR semantic augmentation - variant 1
