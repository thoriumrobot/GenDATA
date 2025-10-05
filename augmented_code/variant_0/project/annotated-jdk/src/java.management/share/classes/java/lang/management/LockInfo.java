/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 2005, 2019, Oracle and/or its affiliates. All rights reserved.
    @DO * Positive NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @This * Positive code is free software; you can redistribute it and/or modify it
    @under * Positive the terms of the GNU General Public License version 2 only, as
    @published * Positive by the Free Software Foundation.  Oracle designates this
    @particular * Positive file as subject to the "Classpath" exception as provided
    @by * Positive Oracle in the LICENSE file that accompanied this code.
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
 */
    @Positive
package java.lang.management;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.checker.signature.qual.ClassGetName;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import javax.management.openmbean.CompositeData;
    @Positive
import java.util.concurrent.locks.*;
    @Positive
import sun.management.LockInfoCompositeData;

    @Positive
@AnnotatedFor({ "signature", "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public class LockInfo {

    @Positive
    public LockInfo(@ClassGetName String className, int identityHashCode) {
    @Positive
    }

    @Positive
    @ClassGetName
    @Positive
    public String getClassName();

    @Positive
    public int getIdentityHashCode();

    @Positive
    public static LockInfo from(CompositeData cd);

    @Positive
    public String toString();
    @Positive
}
