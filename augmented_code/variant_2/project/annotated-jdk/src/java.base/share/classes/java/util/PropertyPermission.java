/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 1997, 2019, Oracle and/or its affiliates. All rights reserved.
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
package java.util;

    @Positive
import org.checkerframework.checker.lock.qual.GuardSatisfied;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.IOException;
    @Positive
import java.io.ObjectInputStream;
    @Positive
import java.io.ObjectOutputStream;
    @Positive
import java.io.ObjectStreamField;
    @Positive
import java.io.Serializable;
    @Positive
import java.security.*;
    @Positive
import java.util.Enumeration;
    @Positive
import java.util.Hashtable;
    @Positive
import java.util.concurrent.ConcurrentHashMap;
    @Positive
import sun.security.util.SecurityConstants;

    @Positive
@AnnotatedFor({ "lock", "nullness", "index" })
    @Positive
public final class PropertyPermission extends BasicPermission {

    @Positive
    public PropertyPermission(String name, @Nullable String actions) {
    @Positive
    }

    @Positive
    @Override
    @Positive
    public boolean implies(Permission p);

    @Positive
    @Pure
    @Positive
    @Override
    @Positive
    public boolean equals(@GuardSatisfied PropertyPermission this, @GuardSatisfied @Nullable Object obj);

    @Positive
    @Pure
    @Positive
    @Override
    @Positive
    public int hashCode(@GuardSatisfied PropertyPermission this);

    @Positive
    static String getActions(int mask);

    @Positive
    @Override
    @Positive
    public String getActions();

    @Positive
    int getMask();

    @Positive
    @Override
    @Positive
    public PermissionCollection newPermissionCollection();
    @Positive
}

    @Positive
final class PropertyPermissionCollection extends PermissionCollection implements Serializable {

    @Positive
    public PropertyPermissionCollection() {
    @Positive
    }

    @Positive
    @Override
    @Positive
    public void add(Permission permission);

    @Positive
    @Override
    @Positive
    public boolean implies(Permission permission);

    @Positive
    @Override
    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    public Enumeration<Permission> elements();
    @Positive
}
