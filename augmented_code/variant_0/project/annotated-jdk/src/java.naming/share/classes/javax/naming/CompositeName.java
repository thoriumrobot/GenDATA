/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 1999, 2020, Oracle and/or its affiliates. All rights reserved.
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
package javax.naming;

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
import java.util.Enumeration;
    @Positive
import java.util.Properties;

    @Positive
public class CompositeName implements Name {

    @Positive
    protected CompositeName(Enumeration<String> comps) {
    @Positive
    }

    @Positive
    public CompositeName(String n) throws InvalidNameException {
    @Positive
    }

    @Positive
    public CompositeName() {
    @Positive
    }

    @Positive
    public String toString();

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    public int hashCode();

    @Positive
    public int compareTo(Object obj);

    @Positive
    public Object clone();

    @Positive
    public int size();

    @Positive
    public boolean isEmpty();

    @Positive
    public Enumeration<String> getAll();

    @Positive
    public String get(int posn);

    @Positive
    public Name getPrefix(int posn);

    @Positive
    public Name getSuffix(int posn);

    @Positive
    public boolean startsWith(Name n);

    @Positive
    public boolean endsWith(Name n);

    @Positive
    public Name addAll(Name suffix) throws InvalidNameException;

    @Positive
    public Name addAll(int posn, Name n) throws InvalidNameException;

    @Positive
    public Name add(String comp) throws InvalidNameException;

    @Positive
    public Name add(int posn, String comp) throws InvalidNameException;

    @Positive
    public Object remove(int posn) throws InvalidNameException;
    @Positive
}
