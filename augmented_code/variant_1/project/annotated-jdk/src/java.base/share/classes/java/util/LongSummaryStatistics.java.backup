/*
 * Copyright (c) 2012, 2019, Oracle and/or its affiliates. All rights reserved.
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
package java.util;

import org.checkerframework.framework.qual.AnnotatedFor;
import java.util.function.IntConsumer;
import java.util.function.LongConsumer;
import java.util.stream.Collector;

@AnnotatedFor({ "lock", "nullness" })
public class LongSummaryStatistics implements LongConsumer, IntConsumer {

    public LongSummaryStatistics() {
    }

    public LongSummaryStatistics(long count, long min, long max, long sum) throws IllegalArgumentException {
    }

    @Override
    public void accept(int value);

    @Override
    public void accept(long value);

    public void combine(LongSummaryStatistics other);

    public final long getCount();

    public final long getSum();

    public final long getMin();

    public final long getMax();

    public final double getAverage();

    @Override
    public String toString();
}
