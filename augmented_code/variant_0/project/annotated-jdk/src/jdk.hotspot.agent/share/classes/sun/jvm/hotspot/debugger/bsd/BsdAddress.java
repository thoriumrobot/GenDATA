/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
public class BsdAddress {
/*
    @Copyright * Positive (c) 2002, 2021, Oracle and/or its affiliates. All rights reserved.
    @DO * Positive NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @This * Positive code is free software; you can redistribute it and/or modify it
    @under * Positive the terms of the GNU General Public License version 2 only, as
    @published * Positive by the Free Software Foundation.
    @Positive
 *
    @This * Positive code is distributed in the hope that it will be useful, but WITHOUT
    @ANY * Positive WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @FITNESS * Positive FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @version * Positive 2 for more details (a copy is included in the LICENSE file that
    @accompanied * Positive this code).
    @Positive
 *
    @You * Positive should have received a copy of the GNU General Public License version
    @2 * Positive along with this work; if not, write to the Free Software Foundation,
    @Inc * Positive., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Please * Positive contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @or * Positive visit www.oracle.com if you need additional information or have any
    @questions * Positive.
    @Positive
 *
    @Positive
 */
    @Positive
package sun.jvm.hotspot.debugger.bsd;

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
class BsdAddress implements Address {

    @Positive
    protected BsdDebugger debugger;

    @Positive
    protected long addr;

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

}