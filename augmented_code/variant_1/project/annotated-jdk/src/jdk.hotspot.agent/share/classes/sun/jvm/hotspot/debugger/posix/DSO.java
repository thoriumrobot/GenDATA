/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 2001, 2020, Oracle and/or its affiliates. All rights reserved.
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
package sun.jvm.hotspot.debugger.posix;

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
import sun.jvm.hotspot.debugger.cdbg.*;
    @Positive
import sun.jvm.hotspot.utilities.memo.*;

    @Positive
public abstract class DSO implements LoadObject {

    @Positive
    public DSO(String filename, long size, Address relocation) {
    @Positive
    }

    @Positive
    public String getName();

    @Positive
    public Address getBase();

    @Positive
    public long getSize();

    @Positive
    public CDebugInfoDataBase getDebugInfoDataBase() throws DebuggerException;

    @Positive
    public BlockSym debugInfoForPC(Address pc) throws DebuggerException;

    @Positive
    public LineNumberInfo lineNumberForPC(Address pc) throws DebuggerException;

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object o);

    @Positive
    public int hashCode();

    @Positive
    protected abstract Address newAddress(long addr);

    @Positive
    protected abstract long getAddressValue(Address addr);
    @Positive
}
