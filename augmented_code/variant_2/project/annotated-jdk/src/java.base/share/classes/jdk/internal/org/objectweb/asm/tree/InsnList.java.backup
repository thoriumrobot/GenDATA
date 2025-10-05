/*
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
package jdk.internal.org.objectweb.asm.tree;

import org.checkerframework.dataflow.qual.Pure;
import java.util.ListIterator;
import java.util.NoSuchElementException;
import jdk.internal.org.objectweb.asm.MethodVisitor;

public class InsnList implements Iterable<AbstractInsnNode> {

    public int size();

    public AbstractInsnNode getFirst();

    public AbstractInsnNode getLast();

    public AbstractInsnNode get(final int index);

    @Pure
    public boolean contains(final AbstractInsnNode insnNode);

    public int indexOf(final AbstractInsnNode insnNode);

    public void accept(final MethodVisitor methodVisitor);

    @Override
    public ListIterator<AbstractInsnNode> iterator();

    @SuppressWarnings("unchecked")
    public ListIterator<AbstractInsnNode> iterator(final int index);

    public AbstractInsnNode[] toArray();

    public void set(final AbstractInsnNode oldInsnNode, final AbstractInsnNode newInsnNode);

    public void add(final AbstractInsnNode insnNode);

    public void add(final InsnList insnList);

    public void insert(final AbstractInsnNode insnNode);

    public void insert(final InsnList insnList);

    public void insert(final AbstractInsnNode previousInsn, final AbstractInsnNode insnNode);

    public void insert(final AbstractInsnNode previousInsn, final InsnList insnList);

    public void insertBefore(final AbstractInsnNode nextInsn, final AbstractInsnNode insnNode);

    public void insertBefore(final AbstractInsnNode nextInsn, final InsnList insnList);

    public void remove(final AbstractInsnNode insnNode);

    void removeAll(final boolean mark);

    public void clear();

    public void resetLabels();

    @SuppressWarnings("rawtypes")
    private final class InsnListIterator implements ListIterator {

        @Override
        public boolean hasNext();

        @Override
        public Object next();

        @Override
        public void remove();

        @Override
        public boolean hasPrevious();

        @Override
        public Object previous();

        @Override
        public int nextIndex();

        @Override
        public int previousIndex();

        @Override
        public void add(final Object o);

        @Override
        public void set(final Object o);
    }
}
