/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 1999, 2021, Oracle and/or its affiliates. All rights reserved.
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
package javax.management;

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
import com.sun.jmx.mbeanserver.GetPropertyAction;
    @Positive
import com.sun.jmx.mbeanserver.Util;
    @Positive
import java.io.IOException;
    @Positive
import java.io.InvalidObjectException;
    @Positive
import java.io.ObjectInputStream;
    @Positive
import java.io.ObjectOutputStream;
    @Positive
import java.io.ObjectStreamField;
    @Positive
import java.security.AccessController;
    @Positive
import java.util.Arrays;
    @Positive
import java.util.Collections;
    @Positive
import java.util.HashMap;
    @Positive
import java.util.Hashtable;
    @Positive
import java.util.Map;

    @Positive
@SuppressWarnings("serial")
    @Positive
public class ObjectName implements Comparable<ObjectName>, QueryExp {

    @Positive
    private static class Property {

    @Positive
        void setKeyIndex(int key_index);

    @Positive
        String getKeyString(String name);

    @Positive
        String getValueString(String name);
    @Positive
    }

    @Positive
    private static class PatternProperty extends Property {
    @Positive
    }

    @Positive
    public static ObjectName getInstance(String name) throws MalformedObjectNameException, NullPointerException;

    @Positive
    public static ObjectName getInstance(String domain, String key, String value) throws MalformedObjectNameException;

    @Positive
    public static ObjectName getInstance(String domain, Hashtable<String, String> table) throws MalformedObjectNameException;

    @Positive
    public static ObjectName getInstance(ObjectName name);

    @Positive
    public ObjectName(String name) throws MalformedObjectNameException {
    @Positive
    }

    @Positive
    public ObjectName(String domain, String key, String value) throws MalformedObjectNameException {
    @Positive
    }

    @Positive
    public ObjectName(String domain, Hashtable<String, String> table) throws MalformedObjectNameException {
    @Positive
    }

    @Positive
    public boolean isPattern();

    @Positive
    public boolean isDomainPattern();

    @Positive
    public boolean isPropertyPattern();

    @Positive
    public boolean isPropertyListPattern();

    @Positive
    public boolean isPropertyValuePattern();

    @Positive
    public boolean isPropertyValuePattern(String property);

    @Positive
    public String getCanonicalName();

    @Positive
    public String getDomain();

    @Positive
    public String getKeyProperty(String property);

    @Positive
    public Hashtable<String, String> getKeyPropertyList();

    @Positive
    public String getKeyPropertyListString();

    @Positive
    public String getCanonicalKeyPropertyListString();

    @Positive
    @Override
    @Positive
    public String toString();

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object object);

    @Positive
    @Override
    @Positive
    public int hashCode();

    @Positive
    public static String quote(String s);

    @Positive
    public static String unquote(String q);

    @Positive
    public static final ObjectName WILDCARD;

    @Positive
    public boolean apply(ObjectName name);

    @Positive
    public void setMBeanServer(MBeanServer mbs);

    @Positive
    public int compareTo(ObjectName name);
    @Positive
}
