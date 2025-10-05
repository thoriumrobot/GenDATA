/*
 * Copyright (c) 1999, 2017, Oracle and/or its affiliates. All rights reserved.
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
package com.sun.jmx.mbeanserver;

import org.checkerframework.dataflow.qual.Pure;
import com.sun.jmx.defaults.ServiceName;
import static com.sun.jmx.defaults.JmxProperties.MBEANSERVER_LOGGER;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.lang.System.Logger.Level;
import java.util.Map;
import java.util.Set;
import javax.management.DynamicMBean;
import javax.management.InstanceAlreadyExistsException;
import javax.management.InstanceNotFoundException;
import javax.management.ObjectName;
import javax.management.QueryExp;
import javax.management.RuntimeOperationsException;

public class Repository {

    public interface RegistrationContext {

        public void registering();

        public void unregistered();
    }

    private static final class ObjectNamePattern {

        public final ObjectName pattern;

        public ObjectNamePattern(ObjectName pattern) {
        }

        public boolean matchKeys(ObjectName name);
    }

    public Repository(String domain) {
    }

    public Repository(String domain, boolean fairLock) {
    }

    public String[] getDomains();

    public void addMBean(final DynamicMBean object, ObjectName name, final RegistrationContext context) throws InstanceAlreadyExistsException;

    @Pure
    public boolean contains(ObjectName name);

    public DynamicMBean retrieve(ObjectName name);

    public Set<NamedObject> query(ObjectName pattern, QueryExp query);

    public void remove(final ObjectName name, final RegistrationContext context) throws InstanceNotFoundException;

    public Integer getCount();

    public String getDefaultDomain();
}
