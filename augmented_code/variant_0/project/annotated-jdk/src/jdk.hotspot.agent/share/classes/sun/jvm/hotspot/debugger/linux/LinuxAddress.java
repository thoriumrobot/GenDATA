/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 2002, 2021, Oracle and/or its affiliates. All rights reserved.
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
package sun.jvm.hotspot.debugger.linux;

    @Positive
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
    @Positive
import org.checkerframework.checker.nullness.qual.NonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import sun.jvm.hotspot.debugger.*;

    @Positive
public class LinuxAddress implements Address {

    @Positive
    protected LinuxDebugger debugger;

    @Positive
    protected long addr;

    @Positive
    public LinuxAddress(LinuxDebugger debugger, long addr) {
    @Positive
    }

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object arg);

    @Positive
    public int hashCode();

    @Positive
    public String toString();

    @Positive
    public long getCIntegerAt(long offset, long numBytes, boolean isUnsigned) throws UnalignedAddressException, UnmappedAddressException;

    @Positive
    public Address getAddressAt(long offset) throws UnalignedAddressException, UnmappedAddressException;

    @Positive
    public Address getCompOopAddressAt(long offset) throws UnalignedAddressException, UnmappedAddressException;

    @Positive
    public Address getCompKlassAddressAt(long offset) throws UnalignedAddressException, UnmappedAddressException;

    @Positive
    public boolean getJBooleanAt(long offset) throws UnalignedAddressException, UnmappedAddressException;

    @Positive
    public byte getJByteAt(long offset) throws UnalignedAddressException, UnmappedAddressException;

    @Positive
    public char getJCharAt(long offset) throws UnalignedAddressException, UnmappedAddressException;

    @Positive
    public double getJDoubleAt(long offset) throws UnalignedAddressException, UnmappedAddressException;

    @Positive
    public float getJFloatAt(long offset) throws UnalignedAddressException, UnmappedAddressException;

    @Positive
    public int getJIntAt(long offset) throws UnalignedAddressException, UnmappedAddressException;

    @Positive
    public long getJLongAt(long offset) throws UnalignedAddressException, UnmappedAddressException;

    @Positive
    public short getJShortAt(long offset) throws UnalignedAddressException, UnmappedAddressException;

    @Positive
    public OopHandle getOopHandleAt(long offset) throws UnalignedAddressException, UnmappedAddressException, NotInHeapException;

    @Positive
    public OopHandle getCompOopHandleAt(long offset) throws UnalignedAddressException, UnmappedAddressException, NotInHeapException;

    @Positive
    public void setCIntegerAt(long offset, long numBytes, long value);

    @Positive
    public void setAddressAt(long offset, Address value);

    @Positive
    public void setJBooleanAt(long offset, boolean value) throws UnmappedAddressException, UnalignedAddressException;

    @Positive
    public void setJByteAt(long offset, byte value) throws UnmappedAddressException, UnalignedAddressException;

    @Positive
    public void setJCharAt(long offset, char value) throws UnmappedAddressException, UnalignedAddressException;

    @Positive
    public void setJDoubleAt(long offset, double value) throws UnmappedAddressException, UnalignedAddressException;

    @Positive
    public void setJFloatAt(long offset, float value) throws UnmappedAddressException, UnalignedAddressException;

    @Positive
    public void setJIntAt(long offset, int value) throws UnmappedAddressException, UnalignedAddressException;

    @Positive
    public void setJLongAt(long offset, long value) throws UnmappedAddressException, UnalignedAddressException;

    @Positive
    public void setJShortAt(long offset, short value) throws UnmappedAddressException, UnalignedAddressException;

    @Positive
    public void setOopHandleAt(long offset, OopHandle value) throws UnmappedAddressException, UnalignedAddressException;

    @Positive
    public Address addOffsetTo(long offset) throws UnsupportedOperationException;

    @Positive
    public OopHandle addOffsetToAsOopHandle(long offset) throws UnsupportedOperationException;

    @Positive
    public long minus(Address arg);

    @Positive
    public boolean lessThan(Address a);

    @Positive
    public boolean lessThanOrEqual(Address a);

    @Positive
    public boolean greaterThan(Address a);

    @Positive
    public boolean greaterThanOrEqual(Address a);

    @Positive
    public Address andWithMask(long mask) throws UnsupportedOperationException;

    @Positive
    public Address orWithMask(long mask) throws UnsupportedOperationException;

    @Positive
    public Address xorWithMask(long mask) throws UnsupportedOperationException;

    @Positive
    public long asLongValue();

    @Positive
    long getValue();

    @Positive
    public static void main(String[] args);
    @Positive
}
