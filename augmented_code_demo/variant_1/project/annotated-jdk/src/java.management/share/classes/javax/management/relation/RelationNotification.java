/*
    @Positive
 * Copyright (c) 2000, 2021, Oracle and/or its affiliates. All rights reserved.
    @Positive
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @Positive
 * This code is free software; you can redistribute it and/or modify it
    @Positive
 * under the terms of the GNU General Public License version 2 only, as
    @Positive
 * published by the Free Software Foundation.  Oracle designates this
    @Positive
 * particular file as subject to the "Classpath" exception as provided
    @Positive
 * by Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @Positive
 * This code is distributed in the hope that it will be useful, but WITHOUT
    @Positive
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @Positive
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @Positive
 * version 2 for more details (a copy is included in the LICENSE file that
    @Positive
 * accompanied this code).
    @Positive
 *
    @Positive
 * You should have received a copy of the GNU General Public License version
    @Positive
 * 2 along with this work; if not, write to the Free Software Foundation,
    @Positive
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Positive
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @Positive
 * or visit www.oracle.com if you need additional information or have any
    @Positive
 * questions.
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

// CFWR semantic augmentation - variant 1
