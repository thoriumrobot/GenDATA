/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 1997, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.security;

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
import sun.security.util.IOUtils;
    @Positive
import java.io.IOException;
    @Positive
import java.io.ByteArrayInputStream;
    @Positive
import java.security.cert.Certificate;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.Hashtable;
    @Positive
import java.lang.reflect.*;
    @Positive
import java.security.cert.*;
    @Positive
import java.util.List;

    @Positive
public final class UnresolvedPermission extends Permission implements java.io.Serializable {

    @Positive
    public UnresolvedPermission(String type, String name, String actions, java.security.cert.Certificate[] certs) {
    @Positive
    }

    @Positive
    Permission resolve(Permission p, java.security.cert.Certificate[] certs);

    @Positive
    public boolean implies(Permission p);

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    public int hashCode();

    @Positive
    public String getActions();

    @Positive
    public String getUnresolvedType();

    @Positive
    public String getUnresolvedName();

    @Positive
    public String getUnresolvedActions();

    @Positive
    public java.security.cert.Certificate[] getUnresolvedCerts();

    @Positive
    public String toString();

    @Positive
    public PermissionCollection newPermissionCollection();
    @Positive
}
