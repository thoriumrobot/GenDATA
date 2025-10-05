/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 2003, 2021, Oracle and/or its affiliates. All rights reserved.
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
package sun.jvm.hotspot.gc.parallel;

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
public class MutableSpace extends VMObject {

    @Positive
    public MutableSpace(Address addr) {
    @Positive
    }

    @Positive
    public Address bottom();

    @Positive
    public Address end();

    @Positive
    public Address top();

    @Positive
    public long used();

    @Positive
    public long capacity();

    @Positive
    public MemRegion usedRegion();

    @Positive
    public OopHandle bottomAsOopHandle();

    @Positive
    public List<MemRegion> getLiveRegions();

    @Positive
    @Pure
    @Positive
    public boolean contains(Address p);

    @Positive
    public void print();

    @Positive
    public void printOn(PrintStream tty);
    @Positive
}
