/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 2003, 2012, Oracle and/or its affiliates. All rights reserved.
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
package sun.management.counter.perf;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectsOnly;
    @Positive
import sun.management.counter.*;
    @Positive
import java.nio.*;
    @Positive
import java.util.*;
    @Positive
import java.util.regex.*;

    @Positive
public class PerfInstrumentation {

    @Positive
    public PerfInstrumentation(ByteBuffer b) {
    @Positive
    }

    @Positive
    public int getMajorVersion();

    @Positive
    public int getMinorVersion();

    @Positive
    public long getModificationTimeStamp();

    @Positive
    void rewind();

    @Positive
    @Pure
    @Positive
    boolean hasNext();

    @Positive
    @SideEffectsOnly("this")
    @Positive
    Counter getNextCounter();

    @Positive
    public synchronized List<Counter> getAllCounters();

    @Positive
    public synchronized List<Counter> findByPattern(String patternString);
    @Positive
}
