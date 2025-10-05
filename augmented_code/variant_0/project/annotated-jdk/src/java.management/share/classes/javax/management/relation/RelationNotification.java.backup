/*
 * Copyright (c) 2000, 2021, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */
package javax.management.relation;

import org.checkerframework.checker.interning.qual.Interned;
import org.checkerframework.framework.qual.AnnotatedFor;
import javax.management.Notification;
import javax.management.ObjectName;
import java.io.InvalidObjectException;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.ObjectStreamField;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import com.sun.jmx.mbeanserver.GetPropertyAction;
import static com.sun.jmx.mbeanserver.Util.cast;

@AnnotatedFor({ "interning" })
@SuppressWarnings("serial")
public class RelationNotification extends Notification {

    @Interned
    public static final String RELATION_BASIC_CREATION;

    @Interned
    public static final String RELATION_MBEAN_CREATION;

    @Interned
    public static final String RELATION_BASIC_UPDATE;

    @Interned
    public static final String RELATION_MBEAN_UPDATE;

    @Interned
    public static final String RELATION_BASIC_REMOVAL;

    @Interned
    public static final String RELATION_MBEAN_REMOVAL;

    public RelationNotification(String notifType, Object sourceObj, long sequence, long timeStamp, String message, String id, String typeName, ObjectName objectName, List<ObjectName> unregMBeanList) throws IllegalArgumentException {
    }

    public RelationNotification(String notifType, Object sourceObj, long sequence, long timeStamp, String message, String id, String typeName, ObjectName objectName, String name, List<ObjectName> newValue, List<ObjectName> oldValue) throws IllegalArgumentException {
    }

    public String getRelationId();

    public String getRelationTypeName();

    public ObjectName getObjectName();

    public List<ObjectName> getMBeansToUnregister();

    public String getRoleName();

    public List<ObjectName> getOldRoleValue();

    public List<ObjectName> getNewRoleValue();
}
