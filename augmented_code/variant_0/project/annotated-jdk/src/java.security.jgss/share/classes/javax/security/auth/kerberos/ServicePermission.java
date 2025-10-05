/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 2000, 2021, Oracle and/or its affiliates. All rights reserved.
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
package javax.security.auth.kerberos;

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
import java.io.IOException;
    @Positive
import java.io.ObjectInputStream;
    @Positive
import java.io.ObjectOutputStream;
    @Positive
import java.io.ObjectStreamField;
    @Positive
import java.security.Permission;
    @Positive
import java.security.PermissionCollection;
    @Positive
import java.util.*;
    @Positive
import java.util.concurrent.ConcurrentHashMap;

    @Positive
public final class ServicePermission extends Permission implements java.io.Serializable {

    @Positive
    public ServicePermission(String servicePrincipal, String action) {
    @Positive
    }

    @Positive
    @Override
    @Positive
    public boolean implies(Permission p);

    @Positive
    boolean impliesIgnoreMask(ServicePermission p);

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    @Override
    @Positive
    public int hashCode();

    @Positive
    static String getActions(int mask);

    @Positive
    @Override
    @Positive
    public String getActions();

    @Positive
    @Override
    @Positive
    public PermissionCollection newPermissionCollection();

    @Positive
    int getMask();
    @Positive
}

    @Positive
final class KrbServicePermissionCollection extends PermissionCollection implements java.io.Serializable {

    @Positive
    public KrbServicePermissionCollection() {
    @Positive
    }

    @Positive
    @Override
    @Positive
    public boolean implies(Permission permission);

    @Positive
    @Override
    @Positive
    public void add(Permission permission);

    @Positive
    @Override
    @Positive
    public Enumeration<Permission> elements();
    @Positive
}
