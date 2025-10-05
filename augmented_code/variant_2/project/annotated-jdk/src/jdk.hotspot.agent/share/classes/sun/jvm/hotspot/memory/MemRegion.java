/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
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
package sun.jvm.hotspot.memory;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import java.util.*;
    @Positive
import sun.jvm.hotspot.debugger.*;
    @Positive
import sun.jvm.hotspot.runtime.*;
    @Positive
import sun.jvm.hotspot.types.*;
    @Positive
import sun.jvm.hotspot.utilities.*;
    @Positive
import sun.jvm.hotspot.utilities.Observable;
    @Positive
import sun.jvm.hotspot.utilities.Observer;

    @Positive
public class MemRegion implements Cloneable {

    @Positive
    public MemRegion() {
    @Positive
    }

    @Positive
    public MemRegion(Address memRegionAddr) {
    @Positive
    }

    @Positive
    public MemRegion(Address start, long wordSize) {
    @Positive
    }

    @Positive
    public MemRegion(Address start, Address limit) {
    @Positive
    }

    @Positive
    public Object clone();

    @Positive
    public MemRegion copy();

    @Positive
    public MemRegion intersection(MemRegion mr2);

    @Positive
    public MemRegion union(MemRegion mr2);

    @Positive
    public Address start();

    @Positive
    public OopHandle startAsOopHandle();

    @Positive
    public Address end();

    @Positive
    public OopHandle endAsOopHandle();

    @Positive
    public void setStart(Address start);

    @Positive
    public void setEnd(Address end);

    @Positive
    public void setWordSize(long wordSize);

    @Positive
    @Pure
    @Positive
    public boolean contains(MemRegion mr2);

    @Positive
    @Pure
    @Positive
    public boolean contains(Address addr);

    @Positive
    public long byteSize();

    @Positive
    public long wordSize();
    @Positive
}
