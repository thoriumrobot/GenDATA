/*
    @Positive
 * Copyright (c) 2000, 2020, Oracle and/or its affiliates. All rights reserved.
    @Positive
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @Positive
 * This code is free software; you can redistribute it and/or modify it
    @Positive
 * under the terms of the GNU General Public License version 2 only, as
    @Positive
 * published by the Free Software Foundation.
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
 *
    @Positive
 */
    @Positive
package sun.jvm.hotspot.gc.shared;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import java.io.*;
    @Positive
import java.util.*;
    @Positive
import sun.jvm.hotspot.debugger.*;
    @Positive
import sun.jvm.hotspot.memory.*;
    @Positive
import sun.jvm.hotspot.runtime.*;
    @Positive
import sun.jvm.hotspot.types.*;
    @Positive
import sun.jvm.hotspot.utilities.Observable;
    @Positive
import sun.jvm.hotspot.utilities.Observer;

    @Positive
public class ContiguousSpace extends CompactibleSpace implements LiveRegionsProvider {

    @Positive
    public ContiguousSpace(Address addr) {
    @Positive
    }

    @Positive
    public Address top();

    @Positive
    public long capacity();

    @Positive
    public long used();

    @Positive
    public long free();

    @Positive
    public MemRegion usedRegion();

    @Positive
    public List<MemRegion> getLiveRegions();

    @Positive
    @Pure
    @Positive
    public boolean contains(Address p);

    @Positive
    public void printOn(PrintStream tty);
    @Positive
}

// CFWR semantic augmentation - variant 1
