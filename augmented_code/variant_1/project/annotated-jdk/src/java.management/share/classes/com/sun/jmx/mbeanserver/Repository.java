/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1999, 2017, Oracle and/or its affiliates. All rights reserved.
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
package com.sun.jmx.mbeanserver;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import com.sun.jmx.defaults.ServiceName;
    @Positive
import static com.sun.jmx.defaults.JmxProperties.MBEANSERVER_LOGGER;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.Collections;
    @Positive
import java.util.HashMap;
    @Positive
import java.util.HashSet;
    @Positive
import java.util.List;
    @Positive
import java.util.concurrent.locks.ReentrantReadWriteLock;
    @Positive
import java.lang.System.Logger.Level;
    @Positive
import java.util.Map;
    @Positive
import java.util.Set;
    @Positive
import javax.management.DynamicMBean;
    @Positive
import javax.management.InstanceAlreadyExistsException;
    @Positive
import javax.management.InstanceNotFoundException;
    @Positive
import javax.management.ObjectName;
    @Positive
import javax.management.QueryExp;
    @Positive
import javax.management.RuntimeOperationsException;

    @Positive
public class Repository {

    @Positive
    public interface RegistrationContext {

    @Positive
        public void registering();

    @Positive
        public void unregistered();
    @Positive
    }

    @Positive
    private static final class ObjectNamePattern {

    @Positive
        public final ObjectName pattern;

    @Positive
        public ObjectNamePattern(ObjectName pattern) {
    @Positive
        }

    @Positive
        public boolean matchKeys(ObjectName name);
    @Positive
    }

    @Positive
    public Repository(String domain) {
    @Positive
    }

    @Positive
    public Repository(String domain, boolean fairLock) {
    @Positive
    }

    @Positive
    public String[] getDomains();

    @Positive
    public void addMBean(final DynamicMBean object, ObjectName name, final RegistrationContext context) throws InstanceAlreadyExistsException;

    @Positive
    @Pure
    @Positive
    public boolean contains(ObjectName name);

    @Positive
    public DynamicMBean retrieve(ObjectName name);

    @Positive
    public Set<NamedObject> query(ObjectName pattern, QueryExp query);

    @Positive
    public void remove(final ObjectName name, final RegistrationContext context) throws InstanceNotFoundException;

    @Positive
    public Integer getCount();

    @Positive
    public String getDefaultDomain();
    @Positive
}
