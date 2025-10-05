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
package javax.management.relation;

    @Positive
import org.checkerframework.checker.interning.qual.Interned;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import javax.management.Notification;
    @Positive
import javax.management.ObjectName;
    @Positive
import java.io.InvalidObjectException;
    @Positive
import java.io.IOException;
    @Positive
import java.io.ObjectInputStream;
    @Positive
import java.io.ObjectOutputStream;
    @Positive
import java.io.ObjectStreamField;
    @Positive
import java.security.AccessController;
    @Positive
import java.security.PrivilegedAction;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.Arrays;
    @Positive
import java.util.Collections;
    @Positive
import java.util.HashSet;
    @Positive
import java.util.List;
    @Positive
import java.util.Set;
    @Positive
import com.sun.jmx.mbeanserver.GetPropertyAction;
    @Positive
import static com.sun.jmx.mbeanserver.Util.cast;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@SuppressWarnings("serial")
    @Positive
public class RelationNotification extends Notification {

    @Positive
    @Interned
    @Positive
    public static final String RELATION_BASIC_CREATION;

    @Positive
    @Interned
    @Positive
    public static final String RELATION_MBEAN_CREATION;

    @Positive
    @Interned
    @Positive
    public static final String RELATION_BASIC_UPDATE;

    @Positive
    @Interned
    @Positive
    public static final String RELATION_MBEAN_UPDATE;

    @Positive
    @Interned
    @Positive
    public static final String RELATION_BASIC_REMOVAL;

    @Positive
    @Interned
    @Positive
    public static final String RELATION_MBEAN_REMOVAL;

    @Positive
    public RelationNotification(String notifType, Object sourceObj, long sequence, long timeStamp, String message, String id, String typeName, ObjectName objectName, List<ObjectName> unregMBeanList) throws IllegalArgumentException {
    @Positive
    }

    @Positive
    public RelationNotification(String notifType, Object sourceObj, long sequence, long timeStamp, String message, String id, String typeName, ObjectName objectName, String name, List<ObjectName> newValue, List<ObjectName> oldValue) throws IllegalArgumentException {
    @Positive
    }

    @Positive
    public String getRelationId();

    @Positive
    public String getRelationTypeName();

    @Positive
    public ObjectName getObjectName();

    @Positive
    public List<ObjectName> getMBeansToUnregister();

    @Positive
    public String getRoleName();

    @Positive
    public List<ObjectName> getOldRoleValue();

    @Positive
    public List<ObjectName> getNewRoleValue();
    @Positive
}
