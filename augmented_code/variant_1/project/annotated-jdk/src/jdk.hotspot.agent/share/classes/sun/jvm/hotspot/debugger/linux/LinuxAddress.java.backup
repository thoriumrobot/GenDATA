/*
 * Copyright (c) 2002, 2021, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.
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
 *
 */
package sun.jvm.hotspot.debugger.linux;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import sun.jvm.hotspot.debugger.*;

public class LinuxAddress implements Address {

    protected LinuxDebugger debugger;

    protected long addr;

    public LinuxAddress(LinuxDebugger debugger, long addr) {
    }

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object arg);

    public int hashCode();

    public String toString();

    public long getCIntegerAt(long offset, long numBytes, boolean isUnsigned) throws UnalignedAddressException, UnmappedAddressException;

    public Address getAddressAt(long offset) throws UnalignedAddressException, UnmappedAddressException;

    public Address getCompOopAddressAt(long offset) throws UnalignedAddressException, UnmappedAddressException;

    public Address getCompKlassAddressAt(long offset) throws UnalignedAddressException, UnmappedAddressException;

    public boolean getJBooleanAt(long offset) throws UnalignedAddressException, UnmappedAddressException;

    public byte getJByteAt(long offset) throws UnalignedAddressException, UnmappedAddressException;

    public char getJCharAt(long offset) throws UnalignedAddressException, UnmappedAddressException;

    public double getJDoubleAt(long offset) throws UnalignedAddressException, UnmappedAddressException;

    public float getJFloatAt(long offset) throws UnalignedAddressException, UnmappedAddressException;

    public int getJIntAt(long offset) throws UnalignedAddressException, UnmappedAddressException;

    public long getJLongAt(long offset) throws UnalignedAddressException, UnmappedAddressException;

    public short getJShortAt(long offset) throws UnalignedAddressException, UnmappedAddressException;

    public OopHandle getOopHandleAt(long offset) throws UnalignedAddressException, UnmappedAddressException, NotInHeapException;

    public OopHandle getCompOopHandleAt(long offset) throws UnalignedAddressException, UnmappedAddressException, NotInHeapException;

    public void setCIntegerAt(long offset, long numBytes, long value);

    public void setAddressAt(long offset, Address value);

    public void setJBooleanAt(long offset, boolean value) throws UnmappedAddressException, UnalignedAddressException;

    public void setJByteAt(long offset, byte value) throws UnmappedAddressException, UnalignedAddressException;

    public void setJCharAt(long offset, char value) throws UnmappedAddressException, UnalignedAddressException;

    public void setJDoubleAt(long offset, double value) throws UnmappedAddressException, UnalignedAddressException;

    public void setJFloatAt(long offset, float value) throws UnmappedAddressException, UnalignedAddressException;

    public void setJIntAt(long offset, int value) throws UnmappedAddressException, UnalignedAddressException;

    public void setJLongAt(long offset, long value) throws UnmappedAddressException, UnalignedAddressException;

    public void setJShortAt(long offset, short value) throws UnmappedAddressException, UnalignedAddressException;

    public void setOopHandleAt(long offset, OopHandle value) throws UnmappedAddressException, UnalignedAddressException;

    public Address addOffsetTo(long offset) throws UnsupportedOperationException;

    public OopHandle addOffsetToAsOopHandle(long offset) throws UnsupportedOperationException;

    public long minus(Address arg);

    public boolean lessThan(Address a);

    public boolean lessThanOrEqual(Address a);

    public boolean greaterThan(Address a);

    public boolean greaterThanOrEqual(Address a);

    public Address andWithMask(long mask) throws UnsupportedOperationException;

    public Address orWithMask(long mask) throws UnsupportedOperationException;

    public Address xorWithMask(long mask) throws UnsupportedOperationException;

    public long asLongValue();

    long getValue();

    public static void main(String[] args);
}
