/*
    @Positive
 * Copyright (c) 1999, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.util;

    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.interning.qual.*;
    @Positive
import org.checkerframework.checker.nonempty.qual.EnsuresNonEmptyIf;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.util.Date;
    @Positive
import java.util.concurrent.atomic.AtomicInteger;
    @Positive
import java.lang.ref.Cleaner.Cleanable;
    @Positive
import jdk.internal.ref.CleanerFactory;

    @Positive
@AnnotatedFor({ "index", "interning", "lock", "nullness" })
    @Positive
@UsesObjectEquals
    @Positive
public class Timer {

    @Positive
    private static class ThreadReaper implements Runnable {

    @Positive
        public void run();
    @Positive
    }

    @Positive
    public Timer() {
    @Positive
    }

    @Positive
    public Timer(boolean isDaemon) {
    @Positive
    }

    @Positive
    public Timer(String name) {
    @Positive
    }

    @Positive
    public Timer(String name, boolean isDaemon) {
    @Positive
    }

    @Positive
    public void schedule(TimerTask task, long delay);

    @Positive
    public void schedule(TimerTask task, Date time);

    @Positive
    public void schedule(TimerTask task, long delay, long period);

    @Positive
    public void schedule(TimerTask task, Date firstTime, long period);

    @Positive
    public void scheduleAtFixedRate(TimerTask task, long delay, long period);

    @Positive
    public void scheduleAtFixedRate(TimerTask task, Date firstTime, long period);

    @Positive
    public void cancel();

    @Positive
    @NonNegative
    @Positive
    public int purge();
    @Positive
}

    @Positive
class TimerThread extends Thread {

    @Positive
    public void run();
    @Positive
}

    @Positive
class TaskQueue {

    @Positive
    @Pure
    @Positive
    int size();

    @Positive
    void add(TimerTask task);

    @Positive
    TimerTask getMin();

    @Positive
    TimerTask get(int i);

    @Positive
    void removeMin();

    @Positive
    void quickRemove(int i);

    @Positive
    void rescheduleMin(long newTime);

    @Positive
    @Pure
    @Positive
    @EnsuresNonEmptyIf(result = false, expression = "this")
    @Positive
    boolean isEmpty();

    @Positive
    void clear();

    @Positive
    void heapify();
    @Positive
}

// CFWR semantic augmentation - variant 0
